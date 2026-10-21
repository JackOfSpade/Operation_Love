"""Pytest plugin recording actual setup/call/teardown outcomes for one process."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from scripts.ci_test_shards import outcome_from_reports, write_json


_inventory: list[str] = []
_reports: dict[str, list[dict]] = defaultdict(list)


def pytest_collection_finish(session):
    _inventory[:] = sorted(item.nodeid for item in session.items)


def pytest_runtest_logreport(report):
    _reports[report.nodeid].append(
        {
            "when": report.when,
            "outcome": report.outcome,
            "wasxfail": bool(getattr(report, "wasxfail", False)),
            "seconds": report.duration,
        }
    )


def pytest_sessionfinish(session, exitstatus):
    path = Path(os.environ["CI_PYTEST_REPORT_PATH"])
    outcomes = [
        {"nodeid": nodeid, "outcome": outcome_from_reports(reports), "reports": reports}
        for nodeid, reports in sorted(_reports.items())
    ]
    write_json(
        path,
        {
            "collected_nodeids": _inventory,
            "exitstatus": int(exitstatus),
            "outcomes": outcomes,
        },
    )
