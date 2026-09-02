#!/usr/bin/env python3
"""Run every configured pytest module in an isolated serial process."""

from __future__ import annotations

import fnmatch
import signal
import shlex
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path


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
            raise SystemExit(f"Configured pytest test root is not a directory: {configured_root}")
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
        raise SystemExit(f"Full-suite pytest collection failed with exit {result.returncode}")

    nodeids = [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and line.split("::", 1)[0].endswith(".py")
    ]
    if not nodeids:
        sys.stdout.write(result.stdout)
        raise SystemExit("Pytest collection returned no test node IDs")
    return nodeids


def main() -> int:
    modules = configured_modules()
    nodeids = collected_nodeids(modules)
    counts = Counter(nodeid.split("::", 1)[0] for nodeid in nodeids)
    nodeids_by_module = {
        module: [nodeid for nodeid in nodeids if nodeid.split("::", 1)[0] == module]
        for module in modules
    }
    discovered = set(modules)
    collected = set(counts)
    if discovered != collected:
        missing = sorted(discovered - collected)
        unexpected = sorted(collected - discovered)
        raise SystemExit(
            "Pytest module discovery mismatch: "
            f"uncollected={missing or 'none'}, undiscovered={unexpected or 'none'}"
        )

    expected_tests = len(nodeids)
    completed_tests = 0
    print(
        f"CI isolation plan: {len(modules)} modules, {expected_tests} collected tests",
        flush=True,
    )
    oom_fallbacks: list[str] = []
    for index, module in enumerate(modules, start=1):
        print(f"::group::[{index}/{len(modules)}] {module}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-n", "0", module],
            cwd=REPO_ROOT,
            check=False,
        )
        print("::endgroup::", flush=True)
        if result.returncode in (-signal.SIGKILL, 128 + signal.SIGKILL, 137):
            oom_fallbacks.append(module)
            print(
                f"CI isolation warning: {module} was SIGKILLed; "
                "retrying its collected tests one at a time",
                flush=True,
            )
            for node_index, nodeid in enumerate(nodeids_by_module[module], start=1):
                print(
                    f"::group::[{index}/{len(modules)}:{node_index}/"
                    f"{counts[module]}] {nodeid}",
                    flush=True,
                )
                node_result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-n", "0", nodeid],
                    cwd=REPO_ROOT,
                    check=False,
                )
                print("::endgroup::", flush=True)
                if node_result.returncode != 0:
                    print(
                        f"CI isolation failed in {nodeid} (exit {node_result.returncode})",
                        flush=True,
                    )
                    return node_result.returncode
        elif result.returncode != 0:
            print(f"CI isolation failed in {module} (exit {result.returncode})", flush=True)
            return result.returncode
        completed_tests += counts[module]

    if completed_tests != expected_tests:
        raise SystemExit(
            f"Executed-count mismatch: expected {expected_tests}, accounted for {completed_tests}"
        )
    print(
        f"CI isolation summary: {len(modules)} modules, "
        f"{completed_tests}/{expected_tests} collected tests passed; "
        f"SIGKILL fallbacks: {oom_fallbacks or 'none'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
