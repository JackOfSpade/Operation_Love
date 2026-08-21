"""Host-side debug log for the Hinge driver — bounded action and evidence screenshots."""
import json
import os
import stat

import pytest

from operation_love.drivers import debuglog as debuglog_module
from operation_love.drivers.debuglog import HingeDebugLog
from operation_love.private_files import UnsafePrivatePathError


def _recs(d):
    return [json.loads(ln) for ln in (d.dir / "actions.jsonl").read_text().splitlines()]


def test_action_writes_record_and_before_after_shots(tmp_path):
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.action("like", before=b"AAA", after=b"BBB", heart=[937, 1600], rose_modal=True)
    rec = _recs(dl)[0]
    assert rec["action"] == "like" and rec["heart"] == [937, 1600] and rec["rose_modal"] is True
    assert (dl.dir / rec["before"]).read_bytes() == b"AAA"
    assert (dl.dir / rec["after"]).read_bytes() == b"BBB"


def test_reopening_a_run_keeps_existing_screenshots_and_allocates_a_new_name(tmp_path):
    # Driver restarts reuse a run id. The JSONL is append-only, so frame names
    # must be as well; otherwise the second instance overwrites first-run evidence.
    first = HingeDebugLog(str(tmp_path), run_id="r")
    first.action("capture", before=b"FIRST_FRAME")

    second = HingeDebugLog(str(tmp_path), run_id="r")
    second.action("capture", before=b"SECOND_FRAME")

    recs = _recs(second)
    assert len(recs) == 2
    first_name, second_name = (rec["before"] for rec in recs)
    assert first_name != second_name
    assert (second.dir / first_name).read_bytes() == b"FIRST_FRAME"
    assert (second.dir / second_name).read_bytes() == b"SECOND_FRAME"


def test_restart_enforces_the_normal_shot_cap_without_rotating_protected_evidence(tmp_path):
    """The screenshot cap belongs to a run, not merely to one DebugLog instance."""
    first = HingeDebugLog(str(tmp_path), run_id="r", keep_shots=8)
    first.action("recoverable", before=b"KEEP_BEFORE", keep_before=True)
    first.error("unexpected", b"KEEP_ERROR", RuntimeError("boom"))
    for frame in (b"NORMAL_1", b"NORMAL_2", b"NORMAL_3"):
        first.action("capture", before=frame)

    records = _recs(first)
    kept_before = records[0]["before"]
    error_shot = records[1]["screenshot"]
    assert records[0]["kept_before"] == kept_before
    assert "_debuglog_shots" not in records[0]
    assert "_debuglog_shots" not in records[1]
    assert "kept_before" not in records[2]

    # Restart with a smaller cap: recovery immediately trims the oldest normal
    # file, then the next normal capture rotates the next-oldest one.
    second = HingeDebugLog(str(tmp_path), run_id="r", keep_shots=2)
    normal_bytes = {
        path.read_bytes() for path in second.dir.glob("*.png")
        if path.name not in {kept_before, error_shot}
    }
    assert normal_bytes == {b"NORMAL_2", b"NORMAL_3"}

    second.action("capture", before=b"NORMAL_4")
    normal_bytes = {
        path.read_bytes() for path in second.dir.glob("*.png")
        if path.name not in {kept_before, error_shot}
    }
    assert normal_bytes == {b"NORMAL_3", b"NORMAL_4"}
    assert (second.dir / kept_before).read_bytes() == b"KEEP_BEFORE"
    assert (second.dir / error_shot).read_bytes() == b"KEEP_ERROR"


def test_action_can_preserve_an_identity_anchor_shot(tmp_path):
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.action("capture_split", before=b"TOP", anchor=b"ANCHOR", after=b"TRIGGER")

    rec = _recs(dl)[0]
    assert (dl.dir / rec["anchor"]).read_bytes() == b"ANCHOR"


def test_normal_shots_rotate_separately_from_recent_error_evidence(tmp_path):
    dl = HingeDebugLog(str(tmp_path), run_id="r", keep_shots=4)
    dl.error("unexpected", b"CRITICAL", ValueError("boom"))   # error shot — must survive rotation
    for i in range(20):                                        # flood with normal shots
        dl.action("dislike", before=bytes([i]), after=bytes([i + 64]))
    pngs = list(dl.dir.glob("*.png"))
    assert len(pngs) <= 4 + 1                                  # rotating cap + the kept error shot
    err = _recs(dl)[0]
    assert err["action"] == "unexpected" and "boom" in err["error"]
    assert (dl.dir / err["screenshot"]).read_bytes() == b"CRITICAL"   # crucial log preserved


def test_recoverable_action_can_keep_its_raw_evidence(tmp_path):
    dl = HingeDebugLog(str(tmp_path), run_id="r", keep_shots=2)
    dl.action("enumeration_segmentation_fallback", before=b"BAD_FRAME", keep_before=True)
    for i in range(8):
        dl.action("capture", before=bytes([i]))

    rec = _recs(dl)[0]
    assert (dl.dir / rec["before"]).read_bytes() == b"BAD_FRAME"


def test_logging_is_best_effort_no_frame(tmp_path):
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.action("capture", before=None, photos=0)               # no screenshot -> no shot keys
    rec = _recs(dl)[0]
    assert rec["action"] == "capture" and "before" not in rec


def test_action_fields_cannot_override_reserved_audit_identity(tmp_path):
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.action("real_action", action="forged_action", ts="forged timestamp")

    rec = _recs(dl)[0]
    assert rec["action"] == "real_action"
    assert rec["ts"] != "forged timestamp"


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "5"])
def test_keep_shots_requires_an_exact_positive_integer(value, tmp_path):
    with pytest.raises(ValueError, match="keep_shots"):
        HingeDebugLog(str(tmp_path), run_id="r", keep_shots=value)


def test_keep_shots_rejects_values_above_bounded_maximum(tmp_path):
    with pytest.raises(ValueError, match="keep_shots"):
        HingeDebugLog(
            str(tmp_path), run_id="r", keep_shots=debuglog_module._MAX_KEEP_SHOTS + 1)


def test_screenshot_write_failure_is_swallowed_but_record_still_written(tmp_path, monkeypatch):
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    def boom(*_args, **_kwargs):                               # disk dies mid-write
        raise OSError("disk full")
    monkeypatch.setattr(debuglog_module, "write_private_bytes", boom)
    dl.action("like", before=b"PNGDATA")                       # must not raise despite the bad write
    dl.error("unexpected", b"PNGDATA", RuntimeError("x"))      # error shots are best-effort too
    recs = _recs(dl)
    assert recs[0]["action"] == "like" and "before" not in recs[0]   # line written, just no shot key
    assert recs[1]["action"] == "unexpected" and "screenshot" not in recs[1]


def test_jsonl_append_failure_is_swallowed(tmp_path, monkeypatch):
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    def boom(*_args, **_kwargs):                               # actions.jsonl append fails
        raise OSError("disk full")
    monkeypatch.setattr(debuglog_module, "append_private_text", boom)
    dl.action("dislike")                                       # must not raise despite the bad append


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_run_directory_and_all_managed_artifacts_are_private_across_restart(tmp_path):
    base = tmp_path / "debug"
    base.mkdir(mode=0o755)
    base.chmod(0o755)
    first = HingeDebugLog(str(base), run_id="r")
    first.action("capture", before=b"BEFORE", after=b"AFTER")

    run_dir = first.dir
    actions = run_dir / "actions.jsonl"
    managed = list(run_dir.glob("*.png"))
    unrelated = run_dir / "notes.png"
    unrelated.write_bytes(b"not managed by DebugLog")
    run_dir.chmod(0o755)
    actions.chmod(0o644)
    for path in managed:
        path.chmod(0o644)
    unrelated.chmod(0o644)

    HingeDebugLog(str(base), run_id="r")

    assert stat.S_IMODE(base.stat().st_mode) == 0o700
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(actions.stat().st_mode) == 0o600
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in managed)
    assert stat.S_IMODE(unrelated.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name != "posix", reason="symlink and POSIX permission semantics")
def test_existing_actions_symlink_is_rejected_without_touching_its_target(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    unrelated = tmp_path / "unrelated.jsonl"
    unrelated.write_text("private elsewhere\n")
    unrelated.chmod(0o644)
    (run_dir / "actions.jsonl").symlink_to(unrelated)

    with pytest.raises(UnsafePrivatePathError):
        HingeDebugLog(str(tmp_path), run_id="r")

    assert unrelated.read_text() == "private elsewhere\n"
    assert stat.S_IMODE(unrelated.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name != "posix", reason="symlink and POSIX permission semantics")
def test_restart_skips_managed_png_symlink_without_reading_or_chmodding_target(tmp_path):
    first = HingeDebugLog(str(tmp_path), run_id="r")
    unrelated = tmp_path / "unrelated.png"
    unrelated.write_bytes(b"outside profile image")
    unrelated.chmod(0o644)
    (first.dir / "99999_capture_before.png").symlink_to(unrelated)

    second = HingeDebugLog(str(tmp_path), run_id="r")
    second.action("capture", before=b"SAFE")

    assert unrelated.read_bytes() == b"outside profile image"
    assert stat.S_IMODE(unrelated.stat().st_mode) == 0o644


# ── content-dedupe (§ observe_waiting heartbeat wrote ~30MB/run, 46.6% byte-identical) ──────
def test_identical_frames_write_one_png_but_two_jsonl_records(tmp_path):
    """The observe_waiting heartbeat calls .action() with the SAME unchanged screencap bytes
    on every no_change poll. Each call must still get its own actions.jsonl record (so the
    timestamp/reason history is intact) but must not cost a second ~1MB PNG on disk."""
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.action("observe_waiting", after=b"SAME_SCREEN", reason="no_change")
    dl.action("observe_waiting", after=b"SAME_SCREEN", reason="no_change")
    dl.action("observe_waiting", after=b"SAME_SCREEN", reason="no_change")

    recs = _recs(dl)
    assert len(recs) == 3                                      # still one record per poll
    names = {r["after"] for r in recs}
    assert len(names) == 1                                     # but all three point at ONE file
    assert len(list(dl.dir.glob("*.png"))) == 1                # and only one PNG was ever written
    assert (dl.dir / recs[0]["after"]).read_bytes() == b"SAME_SCREEN"


def test_different_frames_each_write_their_own_png(tmp_path):
    """A real screen change (a new card, a scroll) must never be deduped away."""
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.action("observe_waiting", after=b"SCREEN_A", reason="no_change")
    dl.action("observe_waiting", after=b"SCREEN_B", reason="no_change")

    recs = _recs(dl)
    names = {r["after"] for r in recs}
    assert len(names) == 2
    assert len(list(dl.dir.glob("*.png"))) == 2


def test_dedup_survives_interleaving_with_distinct_frames(tmp_path):
    """A<->B<->A must not merge A and B just because they alternate -- only byte-identical
    content ever collapses, regardless of how many other distinct shots sit between repeats."""
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.action("observe_waiting", after=b"SCREEN_A", reason="no_change")
    dl.action("observe_waiting", after=b"SCREEN_B", reason="no_change")
    dl.action("observe_waiting", after=b"SCREEN_A", reason="no_change")   # repeats the FIRST frame

    recs = _recs(dl)
    assert len(list(dl.dir.glob("*.png"))) == 2                # only 2 distinct PNGs on disk
    assert recs[0]["after"] == recs[2]["after"]                # 1st and 3rd record share a file
    assert recs[0]["after"] != recs[1]["after"]


def test_rotated_away_duplicate_is_saved_fresh_not_pointed_at_a_deleted_file(tmp_path):
    """Once a shot's file has been rotated off disk, a later identical frame must NOT keep
    resolving to that now-deleted filename -- it has to be saved fresh, exactly like any other
    frame this run has never seen before (see the comment on hash-entry cleanup in
    _save_shot's rotation loop)."""
    dl = HingeDebugLog(str(tmp_path), run_id="r", keep_shots=2)
    dl.action("dislike", before=b"REPEAT_ME")                  # shot #1 (will rotate away)
    dl.action("dislike", before=b"FILLER_A")                   # shot #2
    dl.action("dislike", before=b"FILLER_B")                   # shot #3 -> rotation drops shot #1

    recs = _recs(dl)
    first_name = recs[0]["before"]
    assert not (dl.dir / first_name).exists()                  # rotated away, as expected

    dl.action("dislike", before=b"REPEAT_ME")                  # same bytes as the rotated shot
    newest = _recs(dl)[-1]["before"]
    assert newest != first_name                                # NOT the dangling old name
    assert (dl.dir / newest).exists()
    assert (dl.dir / newest).read_bytes() == b"REPEAT_ME"


def test_error_shots_are_never_deduped_against_normal_shots(tmp_path):
    """rotate=False (error) shots remain a fresh write and are never consulted
    as a dedup source for normal shots either, so this path stays as unconditional as it was."""
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.error("unexpected", b"SAME_BYTES", ValueError("boom"))
    dl.action("dislike", before=b"SAME_BYTES")                 # byte-identical to the error shot

    recs = _recs(dl)
    assert recs[0]["screenshot"] != recs[1]["before"]          # each got its own file
    assert len(list(dl.dir.glob("*.png"))) == 2


def test_two_identical_error_shots_both_write_their_own_file(tmp_path):
    """Two recent error shots with identical bytes still both land on disk."""
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.error("unexpected", b"CRASH_FRAME", ValueError("boom1"))
    dl.error("unexpected", b"CRASH_FRAME", ValueError("boom2"))

    recs = _recs(dl)
    assert recs[0]["screenshot"] != recs[1]["screenshot"]
    assert len(list(dl.dir.glob("*.png"))) == 2


def test_retained_error_pool_is_bounded_and_restart_keeps_only_newest(
        monkeypatch, tmp_path):
    monkeypatch.setattr(debuglog_module, "_MAX_RETAINED_SHOTS", 2)
    first = HingeDebugLog(str(tmp_path), run_id="r")
    first.error("unexpected", b"ERROR_1", RuntimeError("one"))
    first.error("unexpected", b"ERROR_2", RuntimeError("two"))
    first.error("unexpected", b"ERROR_3", RuntimeError("three"))

    assert {path.read_bytes() for path in first.dir.glob("*.png")} == {b"ERROR_2", b"ERROR_3"}

    second = HingeDebugLog(str(tmp_path), run_id="r")
    assert {path.read_bytes() for path in second.dir.glob("*.png")} == {b"ERROR_2", b"ERROR_3"}
    second.error("unexpected", b"ERROR_4", RuntimeError("four"))
    assert {path.read_bytes() for path in second.dir.glob("*.png")} == {b"ERROR_3", b"ERROR_4"}


def test_dedupe_never_files_one_action_s_frame_under_another_s_name(tmp_path):
    """Dedupe is keyed on (label, bytes), not bytes alone. Keying on bytes alone also collapses
    ACROSS action types, and the filename bakes in whichever action wrote the bytes first — so
    an observe_decision record could point at `00001_observe_waiting_before.png`. The image
    would be correct, but this folder is read by eye during an incident, and a decision frame
    filed under a waiting name reads as a filing bug at the worst possible moment."""
    log = HingeDebugLog(str(tmp_path), run_id="r")
    frame = b"the-same-screen"

    log.action("observe_waiting", before=frame, reason="no_change")
    log.action("observe_decision", before=frame, decision="pass")

    records = _recs(log)
    assert "observe_waiting" in records[0]["before"]
    assert "observe_decision" in records[1]["before"]      # its own name, not the waiting one


def test_dedupe_still_collapses_the_repeat_that_actually_causes_the_bloat(tmp_path):
    """The whole point: the same action repeating on an unchanged screen (the 15s heartbeat)
    writes one file, not one per poll."""
    log = HingeDebugLog(str(tmp_path), run_id="r")
    frame = b"nothing-has-moved"

    for _ in range(12):
        log.action("observe_waiting", before=frame, reason="no_change")

    assert len(list(log.dir.glob("*.png"))) == 1
    records = _recs(log)
    assert len(records) == 12 and len({r["before"] for r in records}) == 1


def test_run_id_must_be_a_single_leaf_and_cannot_escape_base_dir(tmp_path):
    base = tmp_path / "debug"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    outside.chmod(0o755)

    for run_id in ("../outside", "nested/run", r"nested\\run", ".", "..", ""):
        with pytest.raises(ValueError, match="one non-dot path component"):
            HingeDebugLog(str(base), run_id=run_id)

    assert not base.exists()
    assert stat.S_IMODE(outside.stat().st_mode) == 0o755
