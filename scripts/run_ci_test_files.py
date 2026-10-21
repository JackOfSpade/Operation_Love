#!/usr/bin/env python3
"""Run one deterministic shard, keeping each module in an isolated serial process."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import signal
import shlex
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__:
    from .ci_test_shards import (
        SUCCESS_OUTCOMES,
        canonical_inventory,
        inventory_digest,
        outcome_from_reports,
        partition,
        write_json,
    )
else:
    from ci_test_shards import (
        SUCCESS_OUTCOMES,
        canonical_inventory,
        inventory_digest,
        outcome_from_reports,
        partition,
        write_json,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _words(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise SystemExit(f"Unsupported pytest configuration value: {value!r}")


def configured_modules() -> list[str]:
    with PYPROJECT.open("rb") as stream:
        config = tomllib.load(stream)
    pytest_config = config.get("tool", {}).get("pytest", {}).get("ini_options", {})
    roots = _words(pytest_config.get("testpaths"), ("tests",))
    patterns = _words(pytest_config.get("python_files"), ("test_*.py", "*_test.py"))

    modules: set[str] = set()
    for configured_root in roots:
        root = (REPO_ROOT / configured_root).resolve()
        if not root.is_dir():
            raise SystemExit(
                f"Configured pytest test root is not a directory: {configured_root}"
            )
        for candidate in root.rglob("*.py"):
            if candidate.is_file() and any(
                fnmatch.fnmatchcase(candidate.name, pattern) for pattern in patterns
            ):
                modules.add(candidate.relative_to(REPO_ROOT).as_posix())

    if not modules:
        raise SystemExit("No pytest modules found under the configured test roots")
    return sorted(modules)


def collected_nodeids(modules: list[str]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-n",
        "0",
        *modules,
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        raise SystemExit(
            f"Full-suite pytest collection failed with exit {result.returncode}"
        )

    nodeids = [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and line.split("::", 1)[0].endswith(".py")
    ]
    if not nodeids:
        sys.stdout.write(result.stdout)
        raise SystemExit("Pytest collection returned no test node IDs")
    return nodeids


def run_test_process(nodeids: list[str], report_path: Path) -> tuple[int, dict | None]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "-p",
            "scripts.ci_pytest_manifest",
            *nodeids,
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CI_PYTEST_REPORT_PATH": str(report_path)},
        check=False,
    )
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else None
    )
    return result.returncode, report


def verified_outcomes(nodeids: list[str], report: dict | None) -> list[dict]:
    if report is None or report.get("exitstatus") != 0:
        raise ValueError("Successful pytest process did not record a successful report")
    if report.get("collected_nodeids") != sorted(nodeids):
        raise ValueError("Executed pytest collection differs from selected node IDs")
    outcomes = report["outcomes"]
    if sorted(r["nodeid"] for r in outcomes) != sorted(nodeids):
        raise ValueError("Pytest outcomes are missing, duplicated or unexpected")
    for result in outcomes:
        actual = outcome_from_reports(result["reports"])
        if actual not in SUCCESS_OUTCOMES or result.get("outcome") != actual:
            raise ValueError("Pytest recorded a failed or incomplete outcome")
    return outcomes


def run_module(
    nodeids: list[str], directory: Path
) -> tuple[list[dict], list[dict], str | None]:
    code, report = run_test_process(nodeids, directory / "module.json")
    attempts = [{"selected_count": len(nodeids), "returncode": code}]
    if code in (-signal.SIGKILL, 128 + signal.SIGKILL, 137):
        print(
            "CI isolation warning: module was SIGKILLed; retrying selected nodes individually",
            flush=True,
        )
        outcomes = []
        for index, nodeid in enumerate(nodeids):
            print(
                f"::group::SIGKILL fallback [{index + 1}/{len(nodeids)}] {nodeid}",
                flush=True,
            )
            code, report = run_test_process([nodeid], directory / f"node-{index}.json")
            print("::endgroup::", flush=True)
            attempts.append({"selected_count": 1, "returncode": code})
            if code != 0:
                outcomes.extend((report or {}).get("outcomes", []))
                return (
                    outcomes,
                    attempts,
                    f"CI isolation failed in {nodeid} (exit {code})",
                )
            outcomes.extend(verified_outcomes([nodeid], report))
        return outcomes, attempts, None
    if code != 0:
        return (
            (report or {}).get("outcomes", []),
            attempts,
            f"CI isolation failed (exit {code})",
        )
    return verified_outcomes(nodeids, report), attempts, None


def execute_shard(index: int, count: int, manifest: dict, save) -> int:
    modules = configured_modules()
    nodeids = canonical_inventory(collected_nodeids(modules))
    counts = Counter(nodeid.split("::", 1)[0] for nodeid in nodeids)
    discovered = set(modules)
    collected = set(counts)
    if discovered != collected:
        missing = sorted(discovered - collected)
        unexpected = sorted(collected - discovered)
        raise SystemExit(
            "Pytest module discovery mismatch: "
            f"uncollected={missing or 'none'}, undiscovered={unexpected or 'none'}"
        )

    selected = partition(nodeids, count)[index]
    manifest.update(
        inventory=nodeids,
        inventory_digest=inventory_digest(nodeids),
        inventory_count=len(nodeids),
        selected_nodeids=selected,
        selected_count=len(selected),
    )
    save()
    print(
        f"CI isolation plan: {len(modules)} modules, {len(nodeids)} collected tests; "
        f"shard {index + 1}/{count} selects {len(selected)}; inventory {manifest['inventory_digest']}",
        flush=True,
    )
    with TemporaryDirectory(prefix="operation-love-ci-") as temporary:
        for module_index, module in enumerate(modules, start=1):
            subset = [
                nodeid for nodeid in selected if nodeid.split("::", 1)[0] == module
            ]
            if not subset:
                continue
            print(
                f"::group::[{module_index}/{len(modules)}] {module} ({len(subset)} selected)",
                flush=True,
            )
            directory = Path(temporary) / str(module_index)
            directory.mkdir()
            outcomes, attempts, error = run_module(subset, directory)
            manifest["outcomes"].extend(outcomes)
            manifest["attempts"].append({"module": module, "processes": attempts})
            manifest["completed_count"] = len(manifest["outcomes"])
            manifest["outcome_counts"] = dict(
                Counter(r["outcome"] for r in manifest["outcomes"])
            )
            print("::endgroup::", flush=True)
            save()
            if error:
                raise RuntimeError(f"{module}: {error}")
    if manifest["completed_count"] != len(selected):
        raise SystemExit(
            f"Executed-count mismatch: expected {len(selected)}, recorded {manifest['completed_count']}"
        )
    manifest["completed"] = True
    print(
        f"CI isolation summary: {manifest['completed_count']}/{len(selected)} selected tests completed; "
        f"outcomes={manifest['outcome_counts']}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA", "local"))
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Shard index must be in [0, shard count)")
    manifest = {
        "version": 1,
        "source_sha": args.source_sha,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "inventory": [],
        "inventory_digest": inventory_digest([]),
        "inventory_count": 0,
        "selected_nodeids": [],
        "selected_count": 0,
        "outcomes": [],
        "outcome_counts": {},
        "completed_count": 0,
        "attempts": [],
        "completed": False,
        "errors": [],
    }

    def save():
        if args.manifest:
            write_json(args.manifest, manifest)

    try:
        return execute_shard(args.shard_index, args.shard_count, manifest, save)
    except (SystemExit, RuntimeError, ValueError, KeyError, OSError) as error:
        manifest["errors"].append(str(error))
        print(f"CI isolation failed: {error}", flush=True)
        return 1
    finally:
        save()


if __name__ == "__main__":
    raise SystemExit(main())
