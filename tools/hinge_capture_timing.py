"""Attribution reader for the capture timing ledger (see `hinge.HingeDriver._capture_current`'s
and `_index_captured_items`' own TIMING LEDGER paragraphs, and `_scroll_down_one`'s GESTURE
TIMING LEDGER paragraph one level down from those).

WHY THIS TOOL EXISTS. A Hinge profile read is measured live at ~6.0s gesture-to-gesture, but
the read loop's own named primitives (screencap, dwell, gesture, settle, segmentation) only
account for ~4.1-4.6s of that -- roughly 1.4-1.9s per frame, the single largest unexplained cost
in the system, was never measured. `hinge.py` now stamps `time.monotonic()` around every
distinguishable cost in the read loop and in the post-loop item-index fold, and writes one
`capture_iteration_timing` row per frame, one `capture_timing_summary` row per capture, and one
`capture_fold_timing` row per fold into the run's `actions.jsonl` (only when `debug_log` is on).

That first pass found one of the read loop's own buckets, "gesture_s", costing ~3.0s -- 53.6% of
an entire profile read -- with nothing inside it named. A second pass (2026-08-23) instrumented
the gesture itself: `_scroll_down_one`/`_scroll_up_one` and the humanized touch transport they
drive (UHID, and the ADB fallback reachable via `touch_backend: adb`) now write one
`capture_gesture_timing` row per gesture, joinable back to the read-loop iteration it belongs to
via `frame_index` (None for the several gestures issued outside the main read loop -- centering,
re-navigation -- which are still measured, just not attributable to one profile-read frame).

This tool is the other half: it reads that file back and turns the raw rows into a per-bucket
attribution -- totals, per-frame/per-gesture/per-fold medians, and the unattributed share as a
percentage -- so an operator can see exactly where a real run's time went instead of only
knowing that some of it is unaccounted for.

PURE OFFLINE ANALYSIS: no ADB, no screencap, no device, no driver construction. It only reads a
JSONL file already written by a previous (or currently running) debug-logged capture.

    python -m tools.hinge_capture_timing ops/hinge_debug/run_20260823_101500
    python -m tools.hinge_capture_timing ops/hinge_debug/run_20260823_101500/actions.jsonl
    python -m tools.hinge_capture_timing ops/hinge_debug/run_20260823_101500 --json

A bare run directory (as `debug_log` writes it) and a direct path to its `actions.jsonl` are
both accepted; the run directory contains other JSONL-adjacent screenshot files this tool never
touches. Malformed or unrelated JSONL lines (this file also carries every other action this
run logged -- captures, likes, passes, refusals) are silently skipped: this tool cares only
about the four record types named above and is best-effort about everything else, matching the
rest of this diagnostic's own philosophy of never being the thing that breaks a read of a run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ITERATION_ACTION = "capture_iteration_timing"
_SUMMARY_ACTION = "capture_timing_summary"
_FOLD_ACTION = "capture_fold_timing"
_GESTURE_ACTION = "capture_gesture_timing"

# Keys on a raw JSONL record that are bookkeeping, not a timing bucket. Anything else numeric
# left on the record IS a bucket -- this is deliberate: a bucket added to hinge.py's ledger
# later shows up here automatically, with no matching change needed in this reader.
_ITERATION_META_KEYS = frozenset({
    "ts", "action", "frame_index", "exit_reason", "iter_wall_s",
    "before", "after", "anchor", "kept_before"})
_FOLD_META_KEYS = frozenset({
    "ts", "action", "photos", "outcome", "fold_wall_s",
    "before", "after", "anchor", "kept_before"})
_GESTURE_META_KEYS = frozenset({
    "ts", "action", "direction", "frame_index", "gesture_wall_s",
    "before", "after", "anchor", "kept_before"})


@dataclass
class BucketStats:
    """One named cost's contribution across every record it was seen in."""
    total_s: float
    mean_s: float
    median_s: float
    n: int


@dataclass
class TimingAttribution:
    """The full per-bucket breakdown for one phase (the read loop, or the fold)."""
    records: int                       # how many rows this was built from
    wall_s_total: float                # sum of every record's own total (iter_wall_s/fold_wall_s)
    buckets: dict[str, BucketStats] = field(default_factory=dict)

    @property
    def unattributed_s_total(self) -> float:
        stats = self.buckets.get("unattributed_s")
        return stats.total_s if stats is not None else 0.0

    @property
    def unattributed_share_pct(self) -> float:
        if self.wall_s_total <= 0:
            return 0.0
        return 100.0 * self.unattributed_s_total / self.wall_s_total


def parse_actions_jsonl(path: Path) -> list[dict]:
    """Read every well-formed JSON object out of an actions.jsonl file.

    A line that is not valid JSON, or does not decode to a dict, is silently skipped -- this
    file is a shared, append-only log written by many best-effort callers (see debuglog.py's own
    "all methods ... never raise" contract), and a reader over it must be at least as tolerant
    of a truncated or malformed line as the writer already is of everything else.
    """
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _resolve_actions_path(target: Path) -> Path:
    """Accept either a run directory (debuglog.py's own layout) or a direct file path."""
    if target.is_dir():
        candidate = target / "actions.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(f"no actions.jsonl under {target}")
        return candidate
    if not target.exists():
        raise FileNotFoundError(f"no such file: {target}")
    return target


def _summarize(records: list[dict], *, wall_key: str, meta_keys: frozenset[str],
              action: str) -> TimingAttribution:
    rows = [r for r in records if r.get("action") == action]
    wall_s_total = 0.0
    bucket_values: dict[str, list[float]] = {}
    for row in rows:
        wall = row.get(wall_key)
        if isinstance(wall, (int, float)):
            wall_s_total += float(wall)
        for key, value in row.items():
            if key in meta_keys or not isinstance(value, (int, float)):
                continue
            bucket_values.setdefault(key, []).append(float(value))
    buckets = {
        key: BucketStats(
            total_s=sum(values), mean_s=statistics.fmean(values),
            median_s=statistics.median(values), n=len(values))
        for key, values in bucket_values.items()}
    return TimingAttribution(records=len(rows), wall_s_total=wall_s_total, buckets=buckets)


def summarize_iteration_timing(records: list[dict]) -> TimingAttribution:
    """Attribution over every `capture_iteration_timing` row (the read loop, per-frame)."""
    return _summarize(records, wall_key="iter_wall_s", meta_keys=_ITERATION_META_KEYS,
                      action=_ITERATION_ACTION)


def summarize_fold_timing(records: list[dict]) -> TimingAttribution:
    """Attribution over every `capture_fold_timing` row (the post-loop item-index fold)."""
    return _summarize(records, wall_key="fold_wall_s", meta_keys=_FOLD_META_KEYS,
                      action=_FOLD_ACTION)


def summarize_gesture_timing(records: list[dict]) -> TimingAttribution:
    """Attribution over every `capture_gesture_timing` row -- one level down from the read
    loop's own "gesture_s" bucket, decomposing each humanized read-scroll into the named costs
    `_scroll`/`_swipe` and the touch transport (UHID, or the ADB fallback) actually measured.
    """
    return _summarize(records, wall_key="gesture_wall_s", meta_keys=_GESTURE_META_KEYS,
                      action=_GESTURE_ACTION)


def _capture_count(records: list[dict]) -> int:
    """How many capture_current calls this file's summary rows account for.

    A capture that returned early from inside the read loop (Stop, a foreign foreground, an
    open comment sheet) never reaches the summary row (see _capture_current's TIMING LEDGER
    paragraph) -- its iterations are still counted in `summarize_iteration_timing`, just not as
    a "capture" here. This number is therefore a floor on captures attempted, not a count of
    every one of them.
    """
    return sum(1 for r in records if r.get("action") == _SUMMARY_ACTION)


def _bucket_rows(attribution: TimingAttribution) -> list[tuple[str, BucketStats]]:
    """Bucket rows sorted by total descending, `unattributed_s` pinned last -- it is not a cost,
    it is the admission that everything else did not add up to the whole."""
    named = sorted(
        ((k, v) for k, v in attribution.buckets.items() if k != "unattributed_s"),
        key=lambda kv: kv[1].total_s, reverse=True)
    if "unattributed_s" in attribution.buckets:
        named.append(("unattributed_s", attribution.buckets["unattributed_s"]))
    return named


def _format_section(title: str, attribution: TimingAttribution, *, per: str) -> str:
    lines = [title]
    if attribution.records == 0:
        lines.append(f"  (no {per} records found)")
        return "\n".join(lines)
    header = f"  {'bucket':<28}{'total_s':>10}{'mean_s':>10}{'median_s':>10}{'n':>6}{'share':>8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for key, stats in _bucket_rows(attribution):
        share = (100.0 * stats.total_s / attribution.wall_s_total
                 if attribution.wall_s_total > 0 else 0.0)
        lines.append(
            f"  {key:<28}{stats.total_s:>10.3f}{stats.mean_s:>10.3f}"
            f"{stats.median_s:>10.3f}{stats.n:>6d}{share:>7.1f}%")
    lines.append("  " + "-" * (len(header) - 2))
    lines.append(
        f"  {attribution.records} {per}, {attribution.wall_s_total:.3f}s total wall clock, "
        f"unattributed = {attribution.unattributed_share_pct:.1f}%")
    return "\n".join(lines)


def format_report(iteration_records: list[dict], fold_records: list[dict],
                  all_records: list[dict], *, source: str,
                  gesture_records: list[dict] | None = None) -> str:
    iter_attr = summarize_iteration_timing(iteration_records)
    fold_attr = summarize_fold_timing(fold_records)
    gesture_attr = summarize_gesture_timing(gesture_records or [])
    captures = _capture_count(all_records)
    lines = [
        f"Hinge capture timing attribution -- {source}",
        f"captures completed: {captures}   iterations (frames read): {iter_attr.records}   "
        f"gestures: {gesture_attr.records}   folds: {fold_attr.records}",
        "",
        _format_section("READ LOOP (per frame)", iter_attr, per="iterations"),
        "",
        # One level down from READ LOOP's own "gesture_s" bucket: this section is what that
        # bucket is actually made of, per gesture rather than per frame (a frame's read-scroll
        # is one gesture; other call sites -- re-navigation, centering -- add more, with no
        # frame_index of their own, folded into this same attribution).
        _format_section("GESTURE (per read-scroll)", gesture_attr, per="gestures"),
        "",
        _format_section("POST-LOOP FOLD (per capture)", fold_attr, per="folds"),
    ]
    return "\n".join(lines)


def _as_json(iteration_records: list[dict], fold_records: list[dict],
            all_records: list[dict], *, gesture_records: list[dict] | None = None) -> dict:
    iter_attr = summarize_iteration_timing(iteration_records)
    fold_attr = summarize_fold_timing(fold_records)
    gesture_attr = summarize_gesture_timing(gesture_records or [])

    def _dump(attribution: TimingAttribution) -> dict:
        return {
            "records": attribution.records,
            "wall_s_total": round(attribution.wall_s_total, 6),
            "unattributed_share_pct": round(attribution.unattributed_share_pct, 3),
            "buckets": {
                key: {"total_s": round(stats.total_s, 6), "mean_s": round(stats.mean_s, 6),
                      "median_s": round(stats.median_s, 6), "n": stats.n}
                for key, stats in attribution.buckets.items()},
        }

    return {
        "captures": _capture_count(all_records),
        "read_loop": _dump(iter_attr),
        "gesture": _dump(gesture_attr),
        "fold": _dump(fold_attr),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "path", type=Path,
        help="a run's actions.jsonl, or the run directory containing it")
    parser.add_argument(
        "--json", action="store_true",
        help="print the attribution as JSON instead of the terminal table")
    args = parser.parse_args(argv)

    try:
        actions_path = _resolve_actions_path(args.path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    records = parse_actions_jsonl(actions_path)
    iteration_records = [r for r in records if r.get("action") == _ITERATION_ACTION]
    fold_records = [r for r in records if r.get("action") == _FOLD_ACTION]
    gesture_records = [r for r in records if r.get("action") == _GESTURE_ACTION]

    if args.json:
        print(json.dumps(
            _as_json(iteration_records, fold_records, records,
                    gesture_records=gesture_records),
            indent=2))
    else:
        print(format_report(iteration_records, fold_records, records, source=str(actions_path),
                            gesture_records=gesture_records))


if __name__ == "__main__":
    main()
