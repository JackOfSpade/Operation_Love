#!/usr/bin/env python3
"""Fail unless every Linux shard completed exactly its part of the same inventory."""

import argparse
import json
from pathlib import Path

from ci_test_shards import reconcile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    try:
        paths = sorted(args.manifest_dir.rglob("*.json"))
        summary = reconcile(
            [json.loads(p.read_text(encoding="utf-8")) for p in paths],
            args.shard_count,
            args.source_sha,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"CI shard reconciliation failed: {error}") from error
    print("CI shard reconciliation passed: " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
