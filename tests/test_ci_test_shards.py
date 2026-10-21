"""Coverage reconciliation fails closed; sharding retains process isolation and OOM recovery."""

import copy
import json
import os
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

from scripts import run_ci_test_files as driver
from scripts.ci_test_shards import (
    canonical_inventory,
    inventory_digest,
    outcome_from_reports,
    partition,
    reconcile,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_required_workflow_cannot_bypass_shard_reconciliation():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    assert "reconcile-tests" in jobs["required"]["needs"]
    assert jobs["reconcile-tests"]["needs"] == ["test"]
    assert "always()" in jobs["reconcile-tests"]["if"]
    assert jobs["test"]["strategy"]["matrix"]["shard"] == list(range(6))
    assert jobs["test"]["strategy"]["fail-fast"] is False
    uploads = [
        step
        for step in jobs["test"]["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 1 and "always()" in uploads[0]["if"]
    assert uploads[0]["with"]["overwrite"] is True
    assert uploads[0]["with"]["if-no-files-found"] == "error"
    commands = "\n".join(
        step.get("run", "") for step in jobs["reconcile-tests"]["steps"]
    )
    assert "reconcile_ci_test_shards.py" in commands and "--shard-count 6" in commands
    assert "--source-sha '${{ github.sha }}'" in commands


def result(nodeid, outcome="passed"):
    reports = [
        {"when": phase, "outcome": "passed", "wasxfail": False}
        for phase in ("setup", "call", "teardown")
    ]
    if outcome == "skipped":
        reports[1]["outcome"] = "skipped"
    elif outcome == "xfailed":
        reports[1].update(outcome="skipped", wasxfail=True)
    elif outcome == "xpassed":
        reports[1]["wasxfail"] = True
    elif outcome == "failed":
        reports[1]["outcome"] = "failed"
    return {"nodeid": nodeid, "outcome": outcome, "reports": reports}


def process_report(nodeids):
    return {
        "exitstatus": 0,
        "collected_nodeids": sorted(nodeids),
        "outcomes": [result(n) for n in nodeids],
    }


def manifests(count=6):
    inventory = canonical_inventory(
        [f"tests/test_a.py::test_case[value {i}]" for i in range(40)]
    )
    rows = []
    for index, subset in enumerate(partition(inventory, count)):
        outcomes = [result(n) for n in subset]
        rows.append(
            {
                "version": 1,
                "source_sha": "source",
                "shard_index": index,
                "shard_count": count,
                "inventory": inventory.copy(),
                "inventory_digest": inventory_digest(inventory),
                "inventory_count": len(inventory),
                "selected_nodeids": subset.copy(),
                "selected_count": len(subset),
                "outcomes": outcomes,
                "outcome_counts": dict(Counter(r["outcome"] for r in outcomes)),
                "completed_count": len(outcomes),
                "completed": True,
                "errors": [],
            }
        )
    return rows


def test_partition_is_order_independent_disjoint_and_keeps_parameter_ids():
    inventory = [
        "tests/test_a.py::test_x[a::b]",
        "tests/test_b.py::test_y[λ space]",
    ] + [f"tests/test_a.py::test_x[{i}]" for i in range(100)]
    shards = partition(inventory, 6)
    assert shards == partition(list(reversed(inventory)), 6)
    assert sorted(n for shard in shards for n in shard) == sorted(inventory)
    assert len({n for shard in shards for n in shard}) == len(inventory)
    assert all(shard == sorted(shard) for shard in shards)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_shard_counts_fail(count):
    with pytest.raises(ValueError):
        partition(["tests/a.py::test_a"], count)
    with pytest.raises(ValueError):
        reconcile([], count)


def test_duplicate_collection_cannot_be_silently_counted_twice():
    with pytest.raises(ValueError, match="Duplicate"):
        partition(["tests/a.py::test_a"] * 2, 6)


def test_reconciliation_accepts_exact_complete_six_shards():
    summary = reconcile(list(reversed(manifests())), 6, "source")
    assert summary["tests"] == 40
    assert summary["shards"] == 6


@pytest.mark.parametrize(
    "corruption",
    [
        "missing-shard",
        "duplicate-shard",
        "wrong-source",
        "changed-inventory",
        "bad-digest",
        "overlap",
        "missing-outcome",
        "duplicate-outcome",
        "failed-outcome",
        "incomplete-report",
        "cancelled",
        "error",
        "wrong-count",
        "wrong-label",
        "wrong-shard-count",
    ],
)
def test_reconciliation_rejects_lost_or_untrustworthy_coverage(corruption):
    rows = copy.deepcopy(manifests())
    row = rows[0]
    if corruption == "missing-shard":
        rows.pop()
    elif corruption == "duplicate-shard":
        rows[-1] = copy.deepcopy(row)
    elif corruption == "wrong-source":
        row["source_sha"] = "other"
    elif corruption == "changed-inventory":
        row["inventory"][0] = "tests/other.py::test_changed"
        row["inventory"].sort()
        row["inventory_digest"] = inventory_digest(row["inventory"])
    elif corruption == "bad-digest":
        row["inventory_digest"] = "0" * 64
    elif corruption == "overlap":
        row["selected_nodeids"].append(rows[1]["selected_nodeids"][0])
    elif corruption == "missing-outcome":
        row["outcomes"].pop()
    elif corruption == "duplicate-outcome":
        row["outcomes"].append(copy.deepcopy(row["outcomes"][0]))
    elif corruption == "failed-outcome":
        row["outcomes"][0] = result(row["outcomes"][0]["nodeid"], "failed")
    elif corruption == "incomplete-report":
        row["outcomes"][0]["reports"].pop()
    elif corruption == "cancelled":
        row["completed"] = False
    elif corruption == "error":
        row["errors"] = ["pytest collection failed"]
    elif corruption == "wrong-count":
        row["completed_count"] += 1
    elif corruption == "wrong-label":
        row["outcomes"][0]["outcome"] = "skipped"
    elif corruption == "wrong-shard-count":
        row["shard_count"] = 5
    with pytest.raises(ValueError):
        reconcile(rows, 6, "source")


@pytest.mark.parametrize("outcome", ["passed", "skipped", "xfailed", "xpassed"])
def test_successful_pytest_semantics_are_preserved(outcome):
    row = manifests(1)[0]
    row["outcomes"] = [result(n, outcome) for n in row["selected_nodeids"]]
    row["outcome_counts"] = {outcome: len(row["outcomes"])}
    validate_manifest(row, 1, "source")
    assert outcome_from_reports(row["outcomes"][0]["reports"]) == outcome


@pytest.mark.parametrize("killed", [-signal.SIGKILL, 137])
def test_oom_retries_only_this_shards_selected_nodes(monkeypatch, tmp_path, killed):
    nodes = ["tests/a.py::test_a", "tests/a.py::test_b[param::value]"]
    calls = []

    def execute(selected, path):
        calls.append(selected.copy())
        if len(calls) == 1:
            return killed, None
        return 0, process_report(selected)

    monkeypatch.setattr(driver, "run_test_process", execute)
    outcomes, attempts, error = driver.run_module(nodes, tmp_path)
    assert calls == [nodes, [nodes[0]], [nodes[1]]]
    assert [r["nodeid"] for r in outcomes] == nodes
    assert [a["returncode"] for a in attempts] == [killed, 0, 0]
    assert error is None


def test_assertion_failure_is_not_retried(monkeypatch, tmp_path):
    nodes = ["tests/a.py::test_a"]
    calls = []

    def execute(selected, path):
        calls.append(selected)
        return 1, {"outcomes": [result(nodes[0], "failed")]}

    monkeypatch.setattr(driver, "run_test_process", execute)
    outcomes, attempts, error = driver.run_module(nodes, tmp_path)
    assert calls == [nodes]
    assert outcomes[0]["outcome"] == "failed"
    assert len(attempts) == 1 and error


def test_oom_fallback_failure_stops_without_skipping_a_failure(monkeypatch, tmp_path):
    nodes = ["tests/a.py::test_a", "tests/a.py::test_b"]
    calls = []

    def execute(selected, path):
        calls.append(selected.copy())
        if len(calls) == 1:
            return 137, None
        return 1, {"outcomes": [result(selected[0], "failed")]}

    monkeypatch.setattr(driver, "run_test_process", execute)
    outcomes, attempts, error = driver.run_module(nodes, tmp_path)
    assert calls == [nodes, [nodes[0]]]
    assert outcomes[0]["outcome"] == "failed"
    assert [a["returncode"] for a in attempts] == [137, 1]
    assert error


@pytest.mark.parametrize(
    "reports",
    [
        [
            {"when": "setup", "outcome": "passed"},
            {"when": "teardown", "outcome": "skipped"},
        ],
        [
            {"when": "setup", "outcome": "skipped"},
            {"when": "call", "outcome": "passed"},
            {"when": "teardown", "outcome": "passed"},
        ],
    ],
)
def test_incomplete_or_impossible_phase_sequences_cannot_pass(reports):
    assert outcome_from_reports(reports) == "incomplete"


def test_zero_exit_without_actual_outcomes_is_not_a_pass():
    with pytest.raises(ValueError):
        driver.verified_outcomes(["tests/a.py::test_a"], None)
    with pytest.raises(ValueError):
        driver.verified_outcomes(
            ["tests/a.py::test_a"],
            {"exitstatus": 0, "collected_nodeids": [], "outcomes": []},
        )


def test_collection_failure_still_writes_failed_shard_manifest(monkeypatch, tmp_path):
    target = tmp_path / "shard.json"
    monkeypatch.setattr(
        sys, "argv", ["run_ci_test_files.py", "--manifest", str(target)]
    )
    monkeypatch.setattr(
        driver,
        "configured_modules",
        lambda: (_ for _ in ()).throw(SystemExit("collection broken")),
    )
    assert driver.main() == 1
    manifest = json.loads(target.read_text())
    assert manifest["completed"] is False
    assert manifest["errors"] == ["collection broken"]


def test_uncollected_configured_module_cannot_disappear(monkeypatch, tmp_path):
    target = tmp_path / "shard.json"
    monkeypatch.setattr(
        sys, "argv", ["run_ci_test_files.py", "--manifest", str(target)]
    )
    monkeypatch.setattr(
        driver, "configured_modules", lambda: ["tests/a.py", "tests/b.py"]
    )
    monkeypatch.setattr(
        driver, "collected_nodeids", lambda modules: ["tests/a.py::test_a"]
    )
    assert driver.main() == 1
    manifest = json.loads(target.read_text())
    assert "uncollected=['tests/b.py']" in manifest["errors"][0]
    assert not manifest["attempts"] and not manifest["completed"]


def make_fixture_repo(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()
    for name in (
        "run_ci_test_files.py",
        "ci_test_shards.py",
        "ci_pytest_manifest.py",
        "reconcile_ci_test_shards.py",
    ):
        (scripts / name).write_text((ROOT / "scripts" / name).read_text())
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\npythonpath = ["."]\n'
    )
    (tests / "test_example.py").write_text("""import pytest
@pytest.mark.parametrize("value", range(12))
def test_parameter(value):
    assert value >= 0
def test_skip():
    pytest.skip("fixture skip")
@pytest.mark.xfail(reason="expected fixture failure")
def test_expected_failure():
    assert False
@pytest.mark.xfail(reason="non-strict existing pytest semantics")
def test_unexpected_pass():
    pass
""")
    return scripts


def test_real_six_shard_driver_and_reconciliation(tmp_path):
    scripts = make_fixture_repo(tmp_path)
    directory = tmp_path / "manifests"
    for index in range(6):
        command = [
            sys.executable,
            str(scripts / "run_ci_test_files.py"),
            "--shard-index",
            str(index),
            "--shard-count",
            "6",
            "--source-sha",
            "fixture",
            "--manifest",
            str(directory / f"shard-{index}.json"),
        ]
        run = subprocess.run(
            command,
            check=False,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTEST_ADDOPTS": ""},
        )
        assert run.returncode == 0, run.stdout + run.stderr
    command = [
        sys.executable,
        str(scripts / "reconcile_ci_test_shards.py"),
        "--manifest-dir",
        str(directory),
        "--shard-count",
        "6",
        "--source-sha",
        "fixture",
    ]
    run = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert '"tests": 15' in run.stdout
    outcomes = Counter(
        r["outcome"]
        for path in directory.glob("*.json")
        for r in json.loads(path.read_text())["outcomes"]
    )
    assert outcomes == {"passed": 12, "skipped": 1, "xfailed": 1, "xpassed": 1}
    (directory / "shard-5.json").unlink()
    run = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert run.returncode != 0 and "Expected all 6" in run.stderr


def test_real_assertion_failure_records_failed_outcome_and_fails_reconciliation(
    tmp_path,
):
    scripts = make_fixture_repo(tmp_path)
    (tmp_path / "tests/test_failure.py").write_text(
        "def test_failure():\n    assert False\n"
    )
    directory = tmp_path / "manifests"
    command = [
        sys.executable,
        str(scripts / "run_ci_test_files.py"),
        "--source-sha",
        "fixture",
        "--manifest",
        str(directory / "shard-0.json"),
    ]
    run = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTEST_ADDOPTS": ""},
    )
    assert run.returncode == 1, run.stdout + run.stderr
    manifest = json.loads((directory / "shard-0.json").read_text())
    assert manifest["outcome_counts"]["failed"] == 1
    assert manifest["completed"] is False and manifest["errors"]
    assert manifest["attempts"][-1]["processes"] == [
        {"selected_count": 1, "returncode": 1}
    ]
    command = [
        sys.executable,
        str(scripts / "reconcile_ci_test_shards.py"),
        "--manifest-dir",
        str(directory),
        "--shard-count",
        "1",
        "--source-sha",
        "fixture",
    ]
    run = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert run.returncode != 0 and "did not complete" in run.stderr
