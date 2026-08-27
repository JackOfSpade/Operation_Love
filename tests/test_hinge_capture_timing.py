"""The capture timing ledger's own arithmetic, and the `tools/hinge_capture_timing.py` reader
that turns a run's `actions.jsonl` into a per-bucket attribution.

See `hinge.AndroidDriver._capture_current`'s and `_index_captured_items`' TIMING LEDGER
paragraphs for why this exists: a live read is measured at ~6.0s gesture-to-gesture, but the
loop's own named primitives only account for ~4.1-4.6s of that, and nobody had ever measured
where the rest went. This file tests the two low-level pieces directly, with a FAKE, fully
controlled `time.monotonic()` -- not a real capture over a synthetic world (that integration
coverage, proving the ledger is actually wired into the real read loop and fold, lives in
tests/test_hinge_item_capture.py alongside the rest of `_capture_current`'s tests) -- because
what needs proving here is the ARITHMETIC: that named buckets plus `unattributed_s` always
reconstruct the measured wall clock exactly, and that time spent outside every named bucket is
genuinely computed into `unattributed_s` rather than silently assumed to be zero.
"""
from __future__ import annotations

import json
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from operation_love.drivers import hinge
from tools import hinge_capture_timing as hct


class _FakeDebugLog:
    """Records every `action(...)` call verbatim, exactly like the real one but with no I/O."""

    def __init__(self):
        self.records: list[dict] = []

    def action(self, name, **fields):
        self.records.append({"action": name, **fields})


class _FakeDriver:
    """The only attribute `_emit_capture_*_timing` touch: `self._dbg`."""

    def __init__(self, dbg):
        self._dbg = dbg


# =====================================================================================
# _time_bucket: the shared primitive
# =====================================================================================

def test_time_bucket_is_a_true_no_op_when_stamps_is_none(monkeypatch):
    calls = []
    monkeypatch.setattr(hinge.time, "monotonic", lambda: calls.append(1) or 0.0)
    with hinge._time_bucket(None, "screencap_s"):
        pass
    assert calls == [], "no debug log means not one time.monotonic() call, not even a cheap one"


def test_time_bucket_records_the_elapsed_wall_clock():
    values = iter([10.0, 12.5])
    with hinge_time_patched(values):
        stamps: dict[str, float] = {}
        with hinge._time_bucket(stamps, "gesture_s"):
            pass
    assert stamps == {"gesture_s": pytest.approx(2.5)}


def test_time_bucket_accumulates_across_repeated_use_of_the_same_key():
    """Identity tracking runs in two separate `if` blocks inside one iteration (see
    _capture_current); both must land in the SAME bucket, summed, not overwrite each other."""
    values = iter([0.0, 1.0, 5.0, 5.5])
    with hinge_time_patched(values):
        stamps: dict[str, float] = {}
        with hinge._time_bucket(stamps, "identity_tracking_s"):
            pass
        with hinge._time_bucket(stamps, "identity_tracking_s"):
            pass
    assert stamps == {"identity_tracking_s": pytest.approx(1.0 + 0.5)}


def test_time_bucket_still_times_a_block_that_raises():
    """A `break`/`return`/raised exception inside the block must still reach the timer's
    `finally` -- the read loop relies on exactly this for a boundary that cuts an iteration
    short (a capture split, a repeated frame, should_stop)."""
    values = iter([0.0, 2.0])
    with hinge_time_patched(values):
        stamps: dict[str, float] = {}
        with pytest.raises(ValueError, match="boom"):
            with hinge._time_bucket(stamps, "identity_tracking_s"):
                raise ValueError("boom")
    assert stamps == {"identity_tracking_s": pytest.approx(2.0)}


class hinge_time_patched:
    """Tiny scoped monkeypatch for `hinge.time.monotonic`, local to this file.

    A full `monkeypatch` fixture works fine inside a test function, but two of the tests above
    need the patch active only around a `with` block rather than for the whole test -- this is
    just that, without pulling in pytest's fixture machinery for a three-line job.
    """

    def __init__(self, values):
        self._values = values
        self._original = None

    def __enter__(self):
        self._original = hinge.time.monotonic
        hinge.time.monotonic = lambda: next(self._values)
        return self

    def __exit__(self, *exc):
        hinge.time.monotonic = self._original
        return False


# =====================================================================================
# _emit_capture_iteration_timing / _emit_capture_timing_summary
# =====================================================================================

def test_iteration_timing_buckets_and_unattributed_sum_to_the_recorded_wall_clock(monkeypatch):
    dbg = _FakeDebugLog()
    driver = _FakeDriver(dbg)
    monkeypatch.setattr(hinge.time, "monotonic", lambda: 7.0)
    stamps = {"screencap_s": 1.0, "gesture_s": 0.5, "settle_s": 0.25}

    iter_wall_s, unattributed_s = hinge.AndroidDriver._emit_capture_iteration_timing(
        driver, 3, 2.0, dict(stamps), exit_reason="completed")

    assert iter_wall_s == pytest.approx(5.0)                          # 7.0 - iter_start(2.0)
    assert unattributed_s == pytest.approx(5.0 - sum(stamps.values()))
    assert sum(stamps.values()) + unattributed_s == pytest.approx(iter_wall_s)
    assert len(dbg.records) == 1
    record = dbg.records[0]
    assert record["action"] == "capture_iteration_timing"
    assert record["frame_index"] == 3
    assert record["exit_reason"] == "completed"
    assert record["iter_wall_s"] == pytest.approx(5.0)
    assert record["unattributed_s"] == pytest.approx(unattributed_s)
    for key, value in stamps.items():
        assert record[key] == pytest.approx(value)


def test_unmeasured_work_lands_in_unattributed_not_a_named_bucket():
    """The exact shape of the bug this ledger exists to catch: work that happens between two
    named costs, wrapped in neither, must show up as unattributed rather than being silently
    folded into whichever bucket happened to run last (or hidden by an assumed-zero residual).
    """
    dbg = _FakeDebugLog()
    driver = _FakeDriver(dbg)
    # 1.0 -> 3.0: the ONE stubbed, wrapped call (2.0s, attributed to "screencap_s").
    # 3.0 -> 7.0: a deliberate 4.0s gap with NO `_time_bucket` around it at all -- standing in
    # for "a stubbed call that quietly did unmeasured work" (e.g. a retry loop, a blocking
    # syscall the ledger does not yet name).
    values = iter([1.0, 3.0, 7.0])
    with hinge_time_patched(values):
        stamps: dict[str, float] = {}
        with hinge._time_bucket(stamps, "screencap_s"):
            pass
        iter_wall_s, unattributed_s = hinge.AndroidDriver._emit_capture_iteration_timing(
            driver, 0, 0.0, stamps, exit_reason="completed")

    assert stamps == {"screencap_s": pytest.approx(2.0)}, (
        "the unmeasured gap must not have been smuggled into the one named bucket")
    assert iter_wall_s == pytest.approx(7.0)
    assert unattributed_s == pytest.approx(5.0)


def test_no_debug_log_iteration_emit_writes_nothing_and_raises_nothing(monkeypatch):
    driver = _FakeDriver(None)
    monkeypatch.setattr(hinge.time, "monotonic", lambda: 5.0)
    iter_wall_s, unattributed_s = hinge.AndroidDriver._emit_capture_iteration_timing(
        driver, 0, 0.0, {"gesture_s": 1.0}, exit_reason="completed")
    assert iter_wall_s == pytest.approx(5.0)
    assert unattributed_s == pytest.approx(4.0)


def test_timing_summary_rolls_up_the_running_totals(monkeypatch):
    dbg = _FakeDebugLog()
    driver = _FakeDriver(dbg)
    hinge.AndroidDriver._emit_capture_timing_summary(
        driver, 5, {"screencap_s": 3.2, "gesture_s": 1.1}, 12.5, 2.4)
    assert len(dbg.records) == 1
    record = dbg.records[0]
    assert record["action"] == "capture_timing_summary"
    assert record["iterations"] == 5
    assert record["iter_wall_s_total"] == pytest.approx(12.5)
    assert record["unattributed_s_total"] == pytest.approx(2.4)
    assert record["screencap_s"] == pytest.approx(3.2)
    assert record["gesture_s"] == pytest.approx(1.1)


def test_no_debug_log_timing_summary_writes_nothing_and_raises_nothing():
    driver = _FakeDriver(None)
    hinge.AndroidDriver._emit_capture_timing_summary(driver, 5, {"screencap_s": 3.2}, 12.5, 2.4)
    # No assertion beyond "did not raise" -- there is no debug log to inspect.


# =====================================================================================
# _emit_capture_fold_timing (the post-loop fold's own, single-row ledger)
# =====================================================================================

def test_fold_timing_buckets_and_unattributed_sum_to_the_recorded_wall_clock(monkeypatch):
    dbg = _FakeDebugLog()
    driver = _FakeDriver(dbg)
    monkeypatch.setattr(hinge.time, "monotonic", lambda: 4.0)
    stamps = {"item_index_build_s": 1.5, "item_payload_build_s": 0.5}

    hinge.AndroidDriver._emit_capture_fold_timing(
        driver, 1.0, dict(stamps), 36, outcome="usable")

    assert len(dbg.records) == 1
    record = dbg.records[0]
    assert record["action"] == "capture_fold_timing"
    assert record["photos"] == 36
    assert record["outcome"] == "usable"
    assert record["fold_wall_s"] == pytest.approx(3.0)                # 4.0 - fold_start(1.0)
    assert record["unattributed_s"] == pytest.approx(3.0 - sum(stamps.values()))
    for key, value in stamps.items():
        assert record[key] == pytest.approx(value)


def test_no_debug_log_fold_emit_writes_nothing_and_raises_nothing():
    driver = _FakeDriver(None)
    hinge.AndroidDriver._emit_capture_fold_timing(driver, 1.0, {}, 0, outcome="usable")


def test_fold_timing_keeps_legacy_dwell_total_and_emits_an_exact_passive_remainder_split(
        monkeypatch):
    """The C2/C3 fold bucket includes real navigation; its detail must say how much was passive.

    The legacy numeric bucket stays in the row for existing consumers.  The split is nested so
    it cannot be double-counted as two extra fold buckets by generic timing readers, and the
    rounded values reconstruct that legacy total exactly in the JSON-ready record.
    """
    dbg = _FakeDebugLog()
    driver = _FakeDriver(dbg)
    driver._still_photo_dwell_timing_breakdown = {
        "passive_observation_s": 51.8669997,
        "navigation_and_overhead_s": 134.7570003,
    }
    monkeypatch.setattr(hinge.time, "monotonic", lambda: 200.0)

    hinge.AndroidDriver._emit_capture_fold_timing(
        driver, 0.0, {"still_photo_dwell_s": 186.624}, 17, outcome="usable")

    record = dbg.records[0]
    assert record["still_photo_dwell_s"] == pytest.approx(186.624)
    breakdown = record["still_photo_dwell_breakdown"]
    assert breakdown["passive_observation_s"] == pytest.approx(51.867)
    assert breakdown["navigation_and_overhead_s"] == pytest.approx(134.757)
    assert (breakdown["passive_observation_s"] + breakdown["navigation_and_overhead_s"]
            == record["still_photo_dwell_s"])


# =====================================================================================
# _emit_gesture_timing (one level down from _emit_capture_iteration_timing -- see
# _scroll_down_one's GESTURE TIMING LEDGER paragraph for why this exists: the read loop's own
# "gesture_s" bucket turned out to be 53.6% of an entire profile read, 2026-08-23, with nothing
# inside it named)
# =====================================================================================

def test_gesture_timing_buckets_and_unattributed_sum_to_the_recorded_wall_clock(monkeypatch):
    dbg = _FakeDebugLog()
    driver = _FakeDriver(dbg)
    monkeypatch.setattr(hinge.time, "monotonic", lambda: 9.0)
    # No separate "uhid_cleanup_s" (2026-08-24): `hid` and `rm -f` collapsed into one remote
    # `adb shell` round trip, so `uhid_hid_run_s` is the whole compound script's cost now --
    # see uhid.py's `_run_gesture` docstring.
    stamps = {
        "screen_size_s": 0.001, "zone_check_s": 0.002, "foreground_reassert_s": 0.099,
        "uhid_plan_swipe_s": 0.01, "uhid_script_build_s": 0.02,
        "uhid_write_file_s": 0.15, "uhid_hid_run_s": 2.62,
    }

    hinge.AndroidDriver._emit_gesture_timing(
        driver, "down", 6.0, dict(stamps), frame_index=4)

    assert len(dbg.records) == 1
    record = dbg.records[0]
    assert record["action"] == "capture_gesture_timing"
    assert record["direction"] == "down"
    assert record["frame_index"] == 4
    assert record["gesture_wall_s"] == pytest.approx(3.0)              # 9.0 - gesture_start(6.0)
    assert record["unattributed_s"] == pytest.approx(3.0 - sum(stamps.values()))
    assert sum(stamps.values()) + record["unattributed_s"] == pytest.approx(
        record["gesture_wall_s"])
    for key, value in stamps.items():
        assert record[key] == pytest.approx(value)


def test_gesture_timing_frame_index_is_none_outside_the_read_loop(monkeypatch):
    """Re-navigation/centering gestures call `_scroll_down_one`/`_scroll_up_one` with no
    `_iteration` at all -- this is what that looks like on the wire: a real, measured gesture
    row that simply does not claim to belong to any one profile-read frame."""
    dbg = _FakeDebugLog()
    driver = _FakeDriver(dbg)
    monkeypatch.setattr(hinge.time, "monotonic", lambda: 1.0)

    hinge.AndroidDriver._emit_gesture_timing(
        driver, "up", 0.5, {"uhid_hid_run_s": 0.4}, frame_index=None)

    assert dbg.records[0]["frame_index"] is None


def test_gesture_timing_unmeasured_work_lands_in_unattributed_not_a_named_bucket():
    """The exact shape of the bug this ledger exists to catch, one level down from the
    iteration ledger's own version of this test: work inside a gesture that happens between two
    named costs, wrapped in neither (a lock acquire, a Python dispatch, anything this ledger
    does not yet name), must show up as unattributed rather than being silently folded into
    whichever bucket happened to run last."""
    dbg = _FakeDebugLog()
    driver = _FakeDriver(dbg)
    # 1.0 -> 1.2: the ONE stubbed, wrapped call (0.2s, attributed to "uhid_hid_run_s").
    # 1.2 -> 4.0: a deliberate 2.8s gap with NO `_time_bucket` around it at all.
    values = iter([1.0, 1.2, 4.0])
    with hinge_time_patched(values):
        stamps: dict[str, float] = {}
        with hinge._time_bucket(stamps, "uhid_hid_run_s"):
            pass
        hinge.AndroidDriver._emit_gesture_timing(
            driver, "down", 0.0, stamps, frame_index=0)

    assert stamps == {"uhid_hid_run_s": pytest.approx(0.2)}, (
        "the unmeasured gap must not have been smuggled into the one named bucket")
    record = dbg.records[0]
    assert record["gesture_wall_s"] == pytest.approx(4.0)
    assert record["unattributed_s"] == pytest.approx(3.8)


def test_no_debug_log_gesture_emit_writes_nothing_and_raises_nothing(monkeypatch):
    driver = _FakeDriver(None)
    monkeypatch.setattr(hinge.time, "monotonic", lambda: 5.0)
    hinge.AndroidDriver._emit_gesture_timing(
        driver, "down", 2.0, {"uhid_hid_run_s": 1.0}, frame_index=3)
    # No assertion beyond "did not raise" -- there is no debug log to inspect.


# =====================================================================================
# _screencap's own split: "screencap_s" vs "screencap_blank_retry_s"
# =====================================================================================

def _png(fill: int, size: int = 8) -> bytes:
    arr = np.full((size, size), fill, dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


_BLANK = _png(0)
_LIVE = _png(180)


class _FakeAdb:
    def __init__(self, frames):
        self._frames = iter(frames)

    def screencap(self):
        return next(self._frames)


class _FakeScreencapDriver:
    def __init__(self, frames):
        self.adb = _FakeAdb(frames)
        self.targeting_calibration = None

    def _blank_reason(self):
        return "unused"


def test_screencap_timing_splits_the_blank_retry_from_the_screencap_itself(monkeypatch):
    monkeypatch.setattr(hinge.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(hinge, "human_delay", lambda seconds: seconds)
    values = iter([0.0, 0.1,      # first adb.screencap() call: 0.1s
                   1.0, 1.4,      # the blank-retry sleep + second adb.screencap(): 0.4s
                   ])
    monkeypatch.setattr(hinge.time, "monotonic", lambda: next(values))
    driver = _FakeScreencapDriver([_BLANK, _LIVE])
    timing: dict[str, float] = {}

    frame = hinge.AndroidDriver._screencap(driver, _timing=timing)

    assert frame == _LIVE
    assert timing["screencap_s"] == pytest.approx(0.1)
    assert timing["screencap_blank_retry_s"] == pytest.approx(0.4)


def test_screencap_timing_is_none_by_default_for_every_other_caller(monkeypatch):
    """Every one of the ~30 other call sites in hinge.py passes no `_timing` at all -- this is
    the guarantee that costs them nothing, not even a dict write."""
    monkeypatch.setattr(hinge.time, "monotonic", lambda: 0.0)
    driver = _FakeScreencapDriver([_LIVE])
    frame = hinge.AndroidDriver._screencap(driver)
    assert frame == _LIVE


# =====================================================================================
# The reader: tools/hinge_capture_timing.py over a synthetic actions.jsonl
# =====================================================================================

_SYNTHETIC_RECORDS = [
    {"ts": "t0", "action": "capture", "photos": 3},           # an unrelated record type
    {"ts": "t1", "action": "capture_iteration_timing", "frame_index": 0,
     "exit_reason": "completed", "iter_wall_s": 6.0, "unattributed_s": 2.0,
     "screencap_s": 0.6, "read_dwell_s": 1.1, "gesture_s": 1.7, "settle_s": 0.4,
     "downsample_s": 0.05, "identity_band_decode_s": 0.05, "identity_tracking_s": 0.03,
     "frame_sig_s": 0.02, "foreground_check_s": 0.05},
    {"ts": "t2", "action": "capture_iteration_timing", "frame_index": 1,
     "exit_reason": "completed", "iter_wall_s": 5.4, "unattributed_s": 1.4,
     "screencap_s": 0.65, "read_dwell_s": 1.1, "gesture_s": 1.65, "settle_s": 0.4,
     "downsample_s": 0.05, "identity_band_decode_s": 0.05, "identity_tracking_s": 0.03,
     "frame_sig_s": 0.02, "foreground_check_s": 0.05},
    {"ts": "t3", "action": "capture_iteration_timing", "frame_index": 2,
     "exit_reason": "repeated_frame", "iter_wall_s": 0.9,
     "unattributed_s": 0.1, "screencap_s": 0.6, "foreground_check_s": 0.05,
     "sheet_probe_s": 0.02, "downsample_s": 0.05, "frame_sig_s": 0.02, "bottom_detect_s": 0.06},
    {"ts": "t4", "action": "capture_timing_summary", "iterations": 3,
     "iter_wall_s_total": pytest.approx(6.0 + 5.4 + 0.9), "unattributed_s_total": 3.5},
    {"ts": "t5", "action": "capture_fold_timing", "photos": 2, "outcome": "usable",
     "fold_wall_s": 2.0, "unattributed_s": 0.3, "video_mute_markers_s": 0.4,
     "item_index_build_s": 1.0, "item_payload_build_s": 0.3},
    # One gesture per frame that scrolled, plus one from OUTSIDE the read loop (re-navigation --
    # frame_index None, see _emit_gesture_timing's own docstring) that still belongs in this
    # same attribution.
    # No "uhid_cleanup_s" (2026-08-24): `hid` and `rm -f` collapsed into one remote `adb shell`
    # round trip, so `uhid_hid_run_s` is the whole compound script's cost -- see uhid.py's
    # `_run_gesture` docstring for why that merge is honest rather than a fake split.
    {"ts": "t1a", "action": "capture_gesture_timing", "direction": "down", "frame_index": 0,
     "gesture_wall_s": 1.7, "unattributed_s": 0.02, "screen_size_s": 0.0005,
     "zone_check_s": 0.001, "foreground_reassert_s": 0.099, "uhid_plan_swipe_s": 0.01,
     "uhid_script_build_s": 0.02, "uhid_write_file_s": 0.15, "uhid_hid_run_s": 1.4},
    {"ts": "t2a", "action": "capture_gesture_timing", "direction": "down", "frame_index": 1,
     "gesture_wall_s": 1.65, "unattributed_s": 0.015, "screen_size_s": 0.0005,
     "zone_check_s": 0.001, "foreground_reassert_s": 0.099, "uhid_plan_swipe_s": 0.01,
     "uhid_script_build_s": 0.02, "uhid_write_file_s": 0.15, "uhid_hid_run_s": 1.344},
    {"ts": "t2b", "action": "capture_gesture_timing", "direction": "up", "frame_index": None,
     "gesture_wall_s": 0.9, "unattributed_s": 0.05, "foreground_reassert_s": 0.1,
     "uhid_hid_run_s": 0.75},
]


def _write_actions_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            # pytest.approx() objects above are only for THIS module's own readability; make the
            # file genuinely numeric before writing it, the way the real debug log always does.
            plain = {k: (v.expected if hasattr(v, "expected") else v) for k, v in record.items()}
            f.write(json.dumps(plain) + "\n")


def test_reader_attributes_a_synthetic_actions_jsonl(tmp_path):
    path = tmp_path / "actions.jsonl"
    _write_actions_jsonl(path, _SYNTHETIC_RECORDS)

    records = hct.parse_actions_jsonl(path)
    iteration_records = [r for r in records if r["action"] == "capture_iteration_timing"]
    fold_records = [r for r in records if r["action"] == "capture_fold_timing"]

    iter_attr = hct.summarize_iteration_timing(iteration_records)
    assert iter_attr.records == 3
    assert iter_attr.wall_s_total == pytest.approx(6.0 + 5.4 + 0.9)
    # screencap_s appears in all three iterations.
    screencap = iter_attr.buckets["screencap_s"]
    assert screencap.n == 3
    assert screencap.total_s == pytest.approx(0.6 + 0.65 + 0.6)
    assert screencap.mean_s == pytest.approx((0.6 + 0.65 + 0.6) / 3)
    assert screencap.median_s == pytest.approx(0.6)
    # read_dwell_s only appears on the two iterations that reached the dwell.
    assert iter_attr.buckets["read_dwell_s"].n == 2
    unattributed = iter_attr.buckets["unattributed_s"]
    assert unattributed.total_s == pytest.approx(2.0 + 1.4 + 0.1)
    assert iter_attr.unattributed_s_total == pytest.approx(3.5)
    assert iter_attr.unattributed_share_pct == pytest.approx(
        100.0 * 3.5 / (6.0 + 5.4 + 0.9))

    fold_attr = hct.summarize_fold_timing(fold_records)
    assert fold_attr.records == 1
    assert fold_attr.wall_s_total == pytest.approx(2.0)
    assert fold_attr.buckets["item_index_build_s"].total_s == pytest.approx(1.0)
    assert fold_attr.unattributed_share_pct == pytest.approx(100.0 * 0.3 / 2.0)

    gesture_records = [r for r in records if r["action"] == "capture_gesture_timing"]
    gesture_attr = hct.summarize_gesture_timing(gesture_records)
    assert gesture_attr.records == 3
    assert gesture_attr.wall_s_total == pytest.approx(1.7 + 1.65 + 0.9)
    # uhid_hid_run_s appears on all three gestures -- the one bucket every gesture in this
    # synthetic run has, real or re-navigation.
    hid_run = gesture_attr.buckets["uhid_hid_run_s"]
    assert hid_run.n == 3
    assert hid_run.total_s == pytest.approx(1.4 + 1.344 + 0.75)
    # screen_size_s/zone_check_s/uhid_plan_swipe_s only appear on the two read-loop gestures --
    # the re-navigation row (frame_index None) never claimed them.
    assert gesture_attr.buckets["screen_size_s"].n == 2
    assert gesture_attr.buckets["uhid_plan_swipe_s"].n == 2
    assert gesture_attr.unattributed_s_total == pytest.approx(0.02 + 0.015 + 0.05)

    assert hct._capture_count(records) == 1


def test_reader_format_report_is_dense_and_names_every_bucket(tmp_path):
    path = tmp_path / "actions.jsonl"
    _write_actions_jsonl(path, _SYNTHETIC_RECORDS)
    records = hct.parse_actions_jsonl(path)
    iteration_records = [r for r in records if r["action"] == "capture_iteration_timing"]
    fold_records = [r for r in records if r["action"] == "capture_fold_timing"]
    gesture_records = [r for r in records if r["action"] == "capture_gesture_timing"]

    report = hct.format_report(iteration_records, fold_records, records, source=str(path),
                               gesture_records=gesture_records)

    assert "READ LOOP" in report
    assert "GESTURE" in report
    assert "POST-LOOP FOLD" in report
    assert "screencap_s" in report
    assert "unattributed_s" in report
    assert "item_index_build_s" in report
    assert "uhid_hid_run_s" in report
    # GESTURE sits between READ LOOP and POST-LOOP FOLD -- it is the further breakdown of READ
    # LOOP's own "gesture_s" bucket, so it reads immediately after the section it decomposes.
    assert report.index("READ LOOP") < report.index("GESTURE") < report.index("POST-LOOP FOLD")
    # unattributed is reported last within each section -- it is the admission that the named
    # buckets did not cover everything, not a cost competing with them for rank.
    gesture_section = report.split("GESTURE")[1].split("POST-LOOP FOLD")[0]
    assert gesture_section.index("unattributed_s") > gesture_section.index("uhid_hid_run_s")


def test_reader_format_report_defaults_gesture_records_to_empty(tmp_path):
    """A caller that never passes `gesture_records` (an older invocation, or a run predating
    this ledger) still gets a report -- an empty GESTURE section, not a TypeError."""
    path = tmp_path / "actions.jsonl"
    _write_actions_jsonl(path, [_SYNTHETIC_RECORDS[1]])   # one lone iteration row, no gestures
    records = hct.parse_actions_jsonl(path)
    iteration_records = [r for r in records if r["action"] == "capture_iteration_timing"]

    report = hct.format_report(iteration_records, [], records, source=str(path))

    assert "GESTURE" in report
    assert "no gestures records found" in report


def test_reader_json_output_matches_the_table(tmp_path):
    path = tmp_path / "actions.jsonl"
    _write_actions_jsonl(path, _SYNTHETIC_RECORDS)
    records = hct.parse_actions_jsonl(path)
    iteration_records = [r for r in records if r["action"] == "capture_iteration_timing"]
    fold_records = [r for r in records if r["action"] == "capture_fold_timing"]
    gesture_records = [r for r in records if r["action"] == "capture_gesture_timing"]

    payload = hct._as_json(iteration_records, fold_records, records,
                           gesture_records=gesture_records)

    assert payload["captures"] == 1
    assert payload["read_loop"]["records"] == 3
    assert payload["read_loop"]["buckets"]["screencap_s"]["n"] == 3
    assert payload["gesture"]["records"] == 3
    assert payload["gesture"]["buckets"]["uhid_hid_run_s"]["n"] == 3
    assert payload["fold"]["records"] == 1
    assert payload["fold"]["buckets"]["item_index_build_s"]["total_s"] == pytest.approx(1.0)


def test_reader_accepts_a_run_directory_or_a_direct_file_path(tmp_path):
    run_dir = tmp_path / "run_20260823_101500"
    run_dir.mkdir()
    _write_actions_jsonl(run_dir / "actions.jsonl", _SYNTHETIC_RECORDS)

    assert hct._resolve_actions_path(run_dir) == run_dir / "actions.jsonl"
    assert hct._resolve_actions_path(run_dir / "actions.jsonl") == run_dir / "actions.jsonl"

    with pytest.raises(FileNotFoundError):
        hct._resolve_actions_path(tmp_path / "no_such_run")


def test_reader_tolerates_malformed_and_unrelated_lines(tmp_path):
    path = tmp_path / "actions.jsonl"
    path.write_text(
        "not json at all\n"
        '"a json string, not an object"\n'
        '{"ts": "t", "action": "like", "before": "x.png"}\n'
        '{"ts": "t", "action": "capture_iteration_timing", "frame_index": 0, '
        '"exit_reason": "completed", "iter_wall_s": 1.0, "unattributed_s": 1.0}\n',
        encoding="utf-8")
    records = hct.parse_actions_jsonl(path)
    assert len(records) == 2  # the "like" row and the one real timing row
    iteration_records = [r for r in records if r["action"] == "capture_iteration_timing"]
    attr = hct.summarize_iteration_timing(iteration_records)
    assert attr.records == 1
    assert attr.wall_s_total == pytest.approx(1.0)


def test_reader_cli_prints_the_table_by_default_and_json_on_flag(tmp_path, capsys):
    path = tmp_path / "actions.jsonl"
    _write_actions_jsonl(path, _SYNTHETIC_RECORDS)

    hct.main([str(path)])
    out = capsys.readouterr().out
    assert "READ LOOP" in out
    assert "GESTURE" in out

    hct.main([str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["read_loop"]["records"] == 3
    assert payload["gesture"]["records"] == 3


def test_reader_cli_errors_cleanly_on_a_missing_run(tmp_path, capsys):
    with pytest.raises(SystemExit):
        hct.main([str(tmp_path / "does_not_exist")])
    assert "error" in capsys.readouterr().err
