"""Host-side debug log for the Hinge driver — bounded action and evidence screenshots."""
import json
import os
import shutil
import stat
import time

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


# --- cross-run retention (2026-08-28) ---------------------------------------------------
# Per-run screenshot caps were always bounded; the NUMBER of runs never was, and
# data/hinge_debug reached 274 runs / 12GB. These pin the narrow candidate rule, because the
# operation is an irreversible rmtree of an operator's diagnostics.

def _run_dir(base, name, *, mtime, actions=True):
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    if actions:
        (d / "actions.jsonl").write_text('{"action":"x"}\n')
    (d / "00001_shot_before.png").write_bytes(b"png")
    import os
    os.utime(d, (mtime, mtime))
    return d


def test_debug_log_keeps_only_the_newest_runs(tmp_path):
    base = tmp_path / "hinge_debug"
    old = [_run_dir(base, f"run_{i:03d}", mtime=1_000 + i) for i in range(5)]

    HingeDebugLog(str(base), run_id="run_current", keep_runs=2)

    survivors = sorted(d.name for d in base.iterdir() if d.is_dir())
    # ``keep_runs`` counts all normal run directories, including the active run.
    assert survivors == ["run_004", "run_current"]
    assert not old[0].exists() and not old[1].exists() and not old[2].exists() and not old[3].exists()


def test_debug_log_never_prunes_the_run_it_is_writing(tmp_path):
    """A RESTART re-enters an existing run id, so the current dir already has actions.jsonl.

    That is the case where the "not one of ours" filter cannot help and the explicit
    current-run check is the only thing standing between a restart and deleting its own
    evidence. Every prior run here is deliberately NEWER, so a naive "keep the newest N"
    would choose the current run for deletion.
    """
    base = tmp_path / "hinge_debug"
    resumed = _run_dir(base, "run_current", mtime=1_000)     # older, and already has a log
    (resumed / "actions.jsonl").write_text('{"action":"from_the_first_attempt"}\n')
    for i in range(4):
        _run_dir(base, f"run_{i:03d}", mtime=9_000_000 + i)

    log = HingeDebugLog(str(base), run_id="run_current", keep_runs=1)
    log.action("probe")

    assert log.dir.exists()
    text = (log.dir / "actions.jsonl").read_text()
    assert "from_the_first_attempt" in text      # the restarted run's history survived
    assert "probe" in text


def test_debug_log_only_prunes_directories_it_wrote(tmp_path):
    """A folder without actions.jsonl is not this logger's; an rmtree must never reach it."""
    base = tmp_path / "hinge_debug"
    _run_dir(base, "run_old", mtime=1_000)
    foreign = base / "operator_notes"          # no actions.jsonl
    foreign.mkdir(parents=True)
    (foreign / "keepme.txt").write_text("do not delete")

    HingeDebugLog(str(base), run_id="run_current", keep_runs=0 + 1)

    assert foreign.exists() and (foreign / "keepme.txt").read_text() == "do not delete"


def test_debug_log_keeps_every_run_when_unset(tmp_path):
    base = tmp_path / "hinge_debug"
    for i in range(6):
        _run_dir(base, f"run_{i:03d}", mtime=1_000 + i)

    HingeDebugLog(str(base), run_id="run_current")      # keep_runs omitted

    assert len([d for d in base.iterdir() if d.is_dir()]) == 7


def test_debug_log_rejects_a_nonsense_keep_runs(tmp_path):
    with pytest.raises(ValueError, match="keep_runs"):
        HingeDebugLog(str(tmp_path / "d"), run_id="r", keep_runs=0)
    with pytest.raises(ValueError, match="keep_runs"):
        HingeDebugLog(str(tmp_path / "d"), run_id="r", keep_runs="5")


def test_debug_log_never_prunes_a_protected_run(tmp_path):
    """Run dirs are cited from OUTSIDE themselves, and the cited one is by definition old.

    config.yaml's observe_release_evidence.production_run_reference is a literal
    data/hinge_debug/<id> path, and ops/release/<id>/ holds that run's signed artifacts.
    Age-based retention would delete exactly that run first.
    """
    base = tmp_path / "hinge_debug"
    _run_dir(base, "release_evidence", mtime=1)          # the OLDEST run on disk
    for i in range(6):
        _run_dir(base, f"run_{i:03d}", mtime=9_000 + i)

    HingeDebugLog(str(base), run_id="run_current", keep_runs=2,
                  protect_runs={"release_evidence"})

    assert (base / "release_evidence").exists()
    assert (base / "release_evidence" / "actions.jsonl").exists()
    # ...and protection does not consume a retention slot: the newest prior ordinary run survives.
    assert (base / "run_005").exists()
    assert not (base / "run_004").exists()
    assert not (base / "run_000").exists()


def test_every_run_directory_a_test_depends_on_is_protected_from_retention():
    """A test fixture must never be prunable by age. This guard is written in hindsight.

    On 2026-08-28 the first cross-run retention pass deleted `run_20260811_011416` and
    `run_20260821_163736`, both referenced by tests as local replay corpora. The tests skip
    gracefully when the data is absent, so the suite stayed green while two real-pixel replays
    were silently destroyed -- including the one its own docstring calls "the only test that
    would have caught the live failure".

    Age is the wrong signal for these: a historical corpus is old BY DEFINITION. This asserts the
    static invariant instead -- anything tests/ or config.yaml names under hinge_debug must be
    listed in debug_protect_runs (or be the derived release-evidence run) -- and it holds whether
    or not the data currently exists, so it would have failed BEFORE that prune.
    """
    import re
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((repo / "config.yaml").read_text())
    hinge = cfg["apps"]["hinge"]
    protected = set(hinge.get("debug_protect_runs") or ())
    release_run = (hinge.get("observe_release_evidence") or {}).get("production_run_id")
    if release_run:
        protected.add(release_run)

    pattern = re.compile(r'hinge_debug["\']?\s*(?:/|,)\s*["\']?([A-Za-z0-9_-]{8,})')
    referenced: dict[str, set[str]] = {}
    sources = list((repo / "tests").rglob("*.py")) + [repo / "config.yaml"]
    for path in sources:
        for match in pattern.finditer(path.read_text(errors="replace")):
            referenced.setdefault(match.group(1), set()).add(path.name)

    # Only runs that ACTUALLY EXIST can be destroyed, and that is precisely the state the
    # 2026-08-28 prune found: present on disk, named by a test, protected by nothing. Synthetic
    # fixture strings that never name a real directory are not a hazard and are skipped.
    debug_dir = repo / hinge.get("debug_dir", "./data/hinge_debug")
    unprotected = {run: sorted(where) for run, where in referenced.items()
                   if run not in protected and (debug_dir / run).is_dir()}
    assert not unprotected, (
        "these run directories exist on disk and are named by tests/config, but are not in "
        f"apps.hinge.debug_protect_runs, so retention will eventually delete them: {unprotected}")


# --- empty run directories (2026-09-16) -------------------------------------------------
# `__init__` creates the run directory before the run logs anything, so a session that starts
# and dies without logging leaves a directory that could NEVER become a retention candidate --
# the actions.jsonl rule above only ever CONSIDERS a directory that already has one. Measured on
# the live tree 2026-09-16: data/hinge_debug held 106 immediate subdirectories, 61 real runs (50
# normal + 11 protected, i.e. pruning of real runs worked exactly as designed) and 45 holding
# nothing at all, the newest from 2026-09-11 -- an ongoing leak, not an old-naming-era relic.

def _empty_run_dir(base, name, *, age_s):
    """A run directory created but never written to, aged `age_s` seconds into the past."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    when = time.time() - age_s
    os.utime(d, (when, when))
    return d


_STALE = debuglog_module._EMPTY_RUN_RECLAIM_AGE_S + 60      # comfortably past the guard
_FRESH = 5                                                  # seconds old: a live logger


def test_reclaimable_empty_predicate_needs_emptiness_AND_age_and_fails_closed(tmp_path):
    """Pin the predicate directly; the behavioural tests alone cannot reach both of its clauses.

    `_trim_old_runs` deletes with `rmdir`, which refuses a non-empty directory on its own. That
    is deliberate defence in depth, but it also MASKS the emptiness clause from every end-to-end
    test: deleting `if any(entry.iterdir())` leaves the whole suite green, because the removal
    then fails with ENOTEMPTY instead of being declined. Verified by mutation on 2026-09-16.
    Both clauses of the decision therefore get pinned here, at the level where each one is the
    only thing being asked.
    """
    stale, fresh = time.time() - _STALE, time.time() - _FRESH
    predicate = debuglog_module._is_reclaimable_empty_run

    empty_stale = tmp_path / "empty_stale"
    empty_stale.mkdir()
    os.utime(empty_stale, (stale, stale))
    assert predicate(empty_stale, time.time()) is True

    empty_fresh = tmp_path / "empty_fresh"
    empty_fresh.mkdir()
    os.utime(empty_fresh, (fresh, fresh))
    assert predicate(empty_fresh, time.time()) is False      # age clause

    for leftover in ("actions.jsonl", "00001_capture_before.png", ".DS_Store"):
        occupied = tmp_path / f"occupied_{leftover}"
        occupied.mkdir()
        (occupied / leftover).write_bytes(b"evidence")
        os.utime(occupied, (stale, stale))
        assert predicate(occupied, time.time()) is False, leftover   # emptiness clause

    holds_a_subdir = tmp_path / "holds_a_subdir"
    (holds_a_subdir / "nested").mkdir(parents=True)
    os.utime(holds_a_subdir, (stale, stale))
    assert predicate(holds_a_subdir, time.time()) is False

    # Fails CLOSED: something we cannot even list is not evidence of emptiness.
    assert predicate(tmp_path / "never_existed", time.time()) is False


def test_debug_log_reclaims_a_stale_empty_run_directory(tmp_path):
    """The leak itself: 0 bytes, no actions.jsonl, therefore previously immortal."""
    base = tmp_path / "hinge_debug"
    leaked = _empty_run_dir(base, "run_20260911_152205", age_s=_STALE)

    HingeDebugLog(str(base), run_id="run_current", keep_runs=50)

    assert not leaked.exists()
    assert (base / "run_current").is_dir()      # the active run is untouched


def test_debug_log_never_reclaims_a_FRESH_empty_run_directory(tmp_path):
    """The concurrency guard.

    A directory that is empty right now may be one ANOTHER process created moments ago and has
    not written its first record to yet. Deleting it would break that logger's later appends,
    and it would do so invisibly -- the writer keeps its own Path and only discovers the loss
    when a flush fails. Emptiness alone is therefore not a licence to delete; staleness is.
    """
    base = tmp_path / "hinge_debug"
    just_started = _empty_run_dir(base, "run_from_another_driver", age_s=_FRESH)

    HingeDebugLog(str(base), run_id="run_current", keep_runs=1)

    assert just_started.is_dir()


def test_debug_log_never_reclaims_the_active_run_when_it_is_empty_and_old(tmp_path):
    """A restart can re-enter a run id whose directory was created but never written.

    That directory is empty AND old -- nothing updates a directory's mtime while it sits unused
    -- so the age guard alone would happily reclaim the run this logger is about to write into.
    The explicit current-run check is what stops it, which is why it runs BEFORE the
    empty-directory branch and not after.
    """
    base = tmp_path / "hinge_debug"
    _empty_run_dir(base, "run_resumed", age_s=_STALE)

    log = HingeDebugLog(str(base), run_id="run_resumed", keep_runs=1)
    log.action("first_record_after_restart")

    assert log.dir.is_dir()
    assert "first_record_after_restart" in (log.dir / "actions.jsonl").read_text()


def test_debug_log_never_reclaims_a_protected_empty_run_directory(tmp_path):
    """Protection is by NAME, and it has to survive the new rule too.

    A protected id can legitimately be an empty directory -- config.yaml lists two ids
    (run_20260811_011416, run_20260821_163736) precisely so that a FUTURE run reusing either is
    never pruned, and such a directory is empty for exactly as long as it takes that run to log
    its first action. Reclaiming it because "empty holds no evidence" would delete the marker
    that exists to stop the deletion.
    """
    base = tmp_path / "hinge_debug"
    protected_empty = _empty_run_dir(base, "run_20260811_011416", age_s=_STALE)

    HingeDebugLog(str(base), run_id="run_current", keep_runs=1,
                  protect_runs={"run_20260811_011416"})

    assert protected_empty.is_dir()


def test_debug_log_never_reclaims_a_stale_foreign_directory_with_contents(tmp_path):
    """"Not one of ours" still means not ours. Only the EMPTY case was ever ambiguous.

    End-to-end only: `rmdir` would decline this directory even if the predicate misjudged it, so
    the emptiness DECISION is pinned in
    test_reclaimable_empty_predicate_needs_emptiness_AND_age_and_fails_closed instead.
    """
    base = tmp_path / "hinge_debug"
    foreign = base / "operator_notes"
    foreign.mkdir(parents=True)
    (foreign / "keepme.txt").write_text("do not delete")
    when = time.time() - _STALE
    os.utime(foreign, (when, when))

    HingeDebugLog(str(base), run_id="run_current", keep_runs=1)

    assert (foreign / "keepme.txt").read_text() == "do not delete"


def test_a_symlinked_run_directory_never_consumes_a_retention_slot(tmp_path):
    """The symlink guard's observable job is the ARITHMETIC, not the delete call.

    Both delete paths already refuse a symlink on their own -- `rmdir` raises ENOTDIR and
    `shutil.rmtree` raises outright -- so a test that only checks "the target survived" passes
    with the guard mutated out and proves nothing (it did exactly that here before mutation
    testing). What the guard uniquely prevents is a foreign tree ENTERING the candidate list: a
    symlink with a recent mtime would take one of the `keep_runs` slots, the rmtree against it
    would fail silently, and a REAL run's evidence would be evicted in its place. That eviction
    is the assertion below.
    """
    base = tmp_path / "hinge_debug"
    outside = tmp_path / "outside_run"
    outside.mkdir()
    (outside / "actions.jsonl").write_text('{"action":"not_ours"}\n')
    older = [_run_dir(base, f"run_{i:03d}", mtime=1_000 + i) for i in range(2)]
    link = base / "run_20260901_000000"
    link.symlink_to(outside, target_is_directory=True)     # newest by mtime, so it would win

    HingeDebugLog(str(base), run_id="run_current", keep_runs=2)

    assert link.is_symlink() and (outside / "actions.jsonl").exists()
    # keep_runs=2 = the active run + ONE prior run, and that slot belongs to a real run.
    assert older[1].exists(), "the symlink took a retention slot from a real run"
    assert not older[0].exists()                           # normal ageing is unchanged


def test_reclaiming_an_empty_directory_uses_rmdir_so_a_late_write_survives(tmp_path,
                                                                          monkeypatch):
    """The last line of the concurrency guard: the emptiness check is not atomic with the delete.

    Between "this directory is empty" and the removal, another process can write its first
    frame. `rmdir` raises ENOTEMPTY and the evidence stays; `rmtree` would destroy it and report
    success. This forces exactly that window by declaring a directory reclaimable while it has
    contents -- which is what the losing side of the race looks like from inside the loop.

    The directory deliberately holds a SCREENSHOT and no actions.jsonl. A first draft gave it an
    actions.jsonl instead, and that test was worthless: the real loop routes anything with an
    actions.jsonl into `candidates` before the empty branch is ever consulted, so the stub below
    was never called and the assertions passed with rmtree in place. Mutation testing caught it.
    """
    base = tmp_path / "hinge_debug"
    racing = base / "run_that_wrote_late"
    racing.mkdir(parents=True)
    (racing / "00001_capture_before.png").write_bytes(b"arrived_during_the_race")
    calls = []

    def racing_predicate(entry, now):
        calls.append(entry.name)
        return entry.name == "run_that_wrote_late"

    monkeypatch.setattr(debuglog_module, "_is_reclaimable_empty_run", racing_predicate)

    HingeDebugLog(str(base), run_id="run_current", keep_runs=50)

    assert "run_that_wrote_late" in calls          # the branch under test was actually reached
    assert racing.is_dir()
    assert (racing / "00001_capture_before.png").read_bytes() == b"arrived_during_the_race"


def test_an_empty_directory_never_consumes_a_retention_slot(tmp_path):
    """Empty directories are reclaimed OUTSIDE the keep_runs slicing, on purpose.

    An empty directory holds no diagnostics, so letting one occupy a retention slot would evict
    a real run's evidence in exchange for nothing -- the slot is the operator's budget for
    things they might need to read.
    """
    base = tmp_path / "hinge_debug"
    real = [_run_dir(base, f"run_{i:03d}", mtime=1_000 + i) for i in range(3)]
    empties = [_empty_run_dir(base, f"run_empty_{i}", age_s=_STALE) for i in range(4)]

    # keep_runs=3 == the active run + the 2 newest prior runs. The 4 empty directories are NOT
    # part of that arithmetic: if they were, every real run here would have been evicted.
    HingeDebugLog(str(base), run_id="run_current", keep_runs=3)

    assert not any(d.exists() for d in empties)
    assert real[2].exists() and real[1].exists()        # the 2 newest real runs kept their slots
    assert not real[0].exists()                         # the oldest real run still ages out
    assert sorted(d.name for d in base.iterdir() if d.is_dir()) == [
        "run_001", "run_002", "run_current"]


# --- what the retention message claims (2026-09-16) --------------------------------------
# The print used to read "keeping {keep_runs} run directory(ies), including the active run",
# i.e. "keeping 50", while 61 real run directories in fact remained. It described the CONFIGURED
# CAP as though it were the outcome -- the defect commit 64e5d6b6 ("Stop the durable records
# from asserting what they never knew") removed from the durable records.

def test_retention_message_reports_what_remains_not_the_configured_cap(tmp_path, monkeypatch,
                                                                       capsys):
    """A failed rmtree is the cleanest separator of "cap" from "outcome".

    Deletion is per-directory best-effort, so `removed` can be less than `doomed`. Here keep_runs
    is 2 but three normal run directories survive, and the operator is told 3 -- the number they
    will actually find on disk -- rather than the setting.
    """
    base = tmp_path / "hinge_debug"
    for i in range(5):
        _run_dir(base, f"run_{i:03d}", mtime=1_000 + i)

    real_rmtree = shutil.rmtree

    def rmtree_that_fails_on_one(path, *args, **kwargs):
        if os.path.basename(str(path)) == "run_000":
            raise PermissionError("simulated: directory busy")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(debuglog_module.shutil, "rmtree", rmtree_that_fails_on_one)

    HingeDebugLog(str(base), run_id="run_current", keep_runs=2)

    out = capsys.readouterr().out
    surviving = sorted(d.name for d in base.iterdir() if d.is_dir())
    assert surviving == ["run_000", "run_004", "run_current"]
    assert "pruned 3 run directory(ies)" in out
    assert "3 run directory(ies) remain, including the active run" in out
    assert str(len(surviving)) == "3"
    # The old wording, which asserted the setting instead of the result.
    assert "keeping 2" not in out


def test_retention_message_counts_protected_runs_separately(tmp_path, capsys):
    """Protected runs remain IN ADDITION to the cap, so folding them in would overstate again.

    Here keep_runs=5 prunes nothing (only 3 prior normal runs exist) and the message is emitted
    solely because empty directories were reclaimed -- a pass the old code printed nothing for
    at all, since it could not see an empty directory as anything.
    """
    base = tmp_path / "hinge_debug"
    for i in range(3):
        _run_dir(base, f"run_{i:03d}", mtime=1_000 + i)
    for name in ("release_evidence", "video_replay"):
        _run_dir(base, name, mtime=1)
    for i in range(4):
        _empty_run_dir(base, f"run_empty_{i}", age_s=_STALE)

    HingeDebugLog(str(base), run_id="run_current", keep_runs=5,
                  protect_runs={"release_evidence", "video_replay"})

    out = capsys.readouterr().out
    assert "reclaimed 4 empty run directory(ies)" in out
    assert "pruned" not in out                      # no real run was eligible
    assert "4 run directory(ies) remain, including the active run" in out
    assert "plus 2 protected run directory(ies) retained regardless of age" in out
    # And the sentence is true of the disk: 3 prior + active = 4 normal, 2 protected besides.
    assert len([d for d in base.iterdir() if d.is_dir()]) == 6


def test_retention_says_nothing_when_it_deleted_nothing(tmp_path, capsys):
    """Silence still means "no diagnostics were destroyed"; the message must not become noise."""
    base = tmp_path / "hinge_debug"
    _run_dir(base, "run_000", mtime=1_000)
    _empty_run_dir(base, "run_fresh_empty", age_s=_FRESH)

    HingeDebugLog(str(base), run_id="run_current", keep_runs=50)

    assert capsys.readouterr().out == ""


# --- what the retention message still could not account for (2026-09-16, same-day review) ---
# The message above reports the OUTCOME rather than the configured cap, but two reviewers
# independently found the same residual gap in it: `1 + len(candidates) - removed` counts only
# the active run plus directories that HAVE an actions.jsonl, so an EMPTY directory the pass
# declined to reclaim was in no clause at all while still sitting on disk. Under-counting a
# surviving directory is the same class of small untruth as quoting the setting, one notch down.
# These pin the third clause, and the one directory that could be counted twice.

def test_retention_message_counts_an_empty_directory_it_declined_to_reclaim(tmp_path, capsys):
    """The concurrency guard spares a fresh empty directory -- and it is STILL THERE afterwards.

    An operator checks a claim like this by listing the directory, which is why the fix counts
    the straggler instead of narrowing the sentence to "run directory(ies) with recorded actions
    remain": no ls tells you which subdirectories contain an actions.jsonl, so the narrowed
    wording would have been true and unusable. The assertion at the bottom is the whole point --
    the clauses of the sentence add up to what the operator can see.
    """
    base = tmp_path / "hinge_debug"
    for i in range(3):
        _run_dir(base, f"run_{i:03d}", mtime=1_000 + i)
    _empty_run_dir(base, "run_from_another_driver", age_s=_FRESH)

    HingeDebugLog(str(base), run_id="run_current", keep_runs=2)

    out = capsys.readouterr().out
    assert "pruned 2 run directory(ies)" in out
    assert "2 run directory(ies) remain, including the active run" in out
    assert "plus 1 empty run directory(ies) left in place" in out
    on_disk = sorted(d.name for d in base.iterdir() if d.is_dir())
    assert on_disk == ["run_002", "run_current", "run_from_another_driver"]
    assert 2 + 1 == len(on_disk), "the reported clauses no longer add up to what ls shows"


def test_retention_message_counts_an_empty_reclaim_that_failed(tmp_path, monkeypatch, capsys):
    """A reclaim that lost the race is also a directory this pass left behind.

    `reclaimed` counts rmdir calls that SUCCEEDED, so an empty directory that grew contents
    between the check and the delete (the window
    test_reclaiming_an_empty_directory_uses_rmdir_so_a_late_write_survives exists for) falls out
    of that count -- and it has no actions.jsonl either, so it would fall out of every other
    count too. Rare, but excluding it would re-open exactly the hole the third clause closes.

    Forced the same way the rmdir-race test forces it: by declaring a directory with contents
    reclaimable, which is what the losing side of the race looks like from inside the loop.
    """
    base = tmp_path / "hinge_debug"
    doomed_empty = _empty_run_dir(base, "run_empty_stale", age_s=_STALE)
    racing = base / "run_that_wrote_late"
    racing.mkdir(parents=True)
    (racing / "00001_capture_before.png").write_bytes(b"arrived_during_the_race")

    real_predicate = debuglog_module._is_reclaimable_empty_run

    def also_the_racing_one(entry, now):
        return entry.name == "run_that_wrote_late" or real_predicate(entry, now)

    monkeypatch.setattr(debuglog_module, "_is_reclaimable_empty_run", also_the_racing_one)

    HingeDebugLog(str(base), run_id="run_current", keep_runs=50)

    out = capsys.readouterr().out
    assert not doomed_empty.exists()                 # this one really was reclaimed
    assert racing.is_dir()                           # this one's rmdir refused, as designed
    assert "reclaimed 1 empty run directory(ies)" in out
    assert "1 run directory(ies) remain, including the active run" in out
    assert "plus 1 empty run directory(ies) left in place" in out
    assert len([d for d in base.iterdir() if d.is_dir()]) == 1 + 1


def test_retention_message_reports_a_protected_active_run_exactly_once(tmp_path, capsys):
    """An active run whose own id is ALSO in protect_runs belongs to one clause, not two.

    The combination is legitimate: config.yaml protects ids precisely so that a future run
    reusing one is never pruned, and one of those runs eventually IS the active run. The
    active-run check runs first and the "1 +" is unconditional, so the normal clause owns it and
    `protected_kept` means "protected runs other than the active one". Counting it in both would
    report 4 directories where ls finds 3 -- an overstatement, where the fix above closed an
    understatement.
    """
    base = tmp_path / "hinge_debug"
    for i in range(3):
        _run_dir(base, f"run_{i:03d}", mtime=1_000 + i)
    _run_dir(base, "release_evidence", mtime=1)          # the oldest, and protected

    HingeDebugLog(str(base), run_id="run_current", keep_runs=2,
                  protect_runs={"run_current", "release_evidence"})

    out = capsys.readouterr().out
    assert "2 run directory(ies) remain, including the active run" in out
    assert "plus 1 protected run directory(ies) retained regardless of age" in out
    assert "plus 2 protected" not in out             # the active run is not counted twice
    on_disk = sorted(d.name for d in base.iterdir() if d.is_dir())
    assert on_disk == ["release_evidence", "run_002", "run_current"]
    assert 2 + 1 == len(on_disk)


# --- the writer survives losing its own directory (2026-09-16) ---------------------------
# The empty-run reclaim above deletes the directory of a logger that has been silent past
# _EMPTY_RUN_RECLAIM_AGE_S. Its docstring argues that window never opens because "the driver
# logs its first capture as soon as it has a frame" -- true of the Hinge driver, false of a
# driver that is constructed and then BLOCKED (paywalled deck, wedged app, waiting on a human).
# The deletion itself is lossless (the directory was empty by construction), so only FUTURE
# evidence is at risk -- and _write is best-effort, so the victim would lose every subsequent
# record in silence. The writer heals rather than the deletion rule narrowing: widening the rule
# restores the 45-directory leak, and a longer age only moves the window.

def test_logger_whose_directory_vanished_recreates_it_and_keeps_writing(tmp_path):
    base = tmp_path / "hinge_debug"
    log = HingeDebugLog(str(base), run_id="run_blocked_on_a_paywall", keep_runs=50)
    run_dir = log.dir

    run_dir.rmdir()                      # exactly what another pass's empty-run reclaim does
    assert not run_dir.exists()

    log.action("deck_unblocked")         # the first record after the loss

    assert run_dir.is_dir()
    assert "deck_unblocked" in (run_dir / "actions.jsonl").read_text()

    # And the run is fully functional again afterwards, including the screenshots that the
    # healing record itself could not save (_save_shot runs before _write inside action()).
    log.action("like", before=b"PNGDATA")
    assert _recs(log)[1]["before"] == "00001_like_before.png"
    assert (run_dir / "00001_like_before.png").read_bytes() == b"PNGDATA"

    if os.name == "posix":
        # Recovery goes through the same 0700/0600 helpers, so a healed run directory is never
        # re-created at the process umask's looser default.
        assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE((run_dir / "actions.jsonl").stat().st_mode) == 0o600


def test_self_healing_gives_up_after_one_attempt_and_still_never_raises(tmp_path, monkeypatch):
    """A second failure must not reach the caller, and must not become a retry storm.

    The `healed` assertion is what keeps this test honest: "it did not raise" passes just as
    happily when the retry branch is never reached at all, so the recovery attempt is counted
    rather than assumed.
    """
    log = HingeDebugLog(str(tmp_path / "hinge_debug"), run_id="r")
    healed = []
    real_ensure = debuglog_module.ensure_private_dir

    def counting_ensure(path, **kwargs):
        healed.append(path)
        return real_ensure(path, **kwargs)

    def boom(*_args, **_kwargs):
        raise OSError("disk full")       # not a missing directory: healing cannot fix this

    monkeypatch.setattr(debuglog_module, "ensure_private_dir", counting_ensure)
    monkeypatch.setattr(debuglog_module, "append_private_text", boom)

    log.action("still_best_effort")      # must not raise

    assert healed == [log.dir]           # exactly one recovery attempt, then it gave up


def test_a_write_that_succeeds_never_re_creates_the_directory_or_doubles_the_record(tmp_path,
                                                                                    monkeypatch):
    """The healthy path must stop at the successful append, because the recovery branch APPENDS.

    Falling through into it after a write that already worked would write every record twice --
    a worse failure than the one being fixed, since a duplicated action trail misleads exactly
    the person reconstructing what the run did.
    """
    log = HingeDebugLog(str(tmp_path / "hinge_debug"), run_id="r")
    healed = []
    real_ensure = debuglog_module.ensure_private_dir

    def counting_ensure(path, **kwargs):
        healed.append(path)
        return real_ensure(path, **kwargs)

    monkeypatch.setattr(debuglog_module, "ensure_private_dir", counting_ensure)

    log.action("ordinary")

    assert healed == []
    assert [r["action"] for r in _recs(log)] == ["ordinary"]


def test_an_unserialisable_field_is_still_swallowed_and_never_triggers_a_heal(tmp_path,
                                                                              monkeypatch):
    """json.dumps moved OUT of the append's try (the retry needs the line), so it needs a guard.

    hinge.py's NAV_CHAIN_BROKEN path (~11045) documents exactly this hazard -- raw PNG bytes
    reaching action(**fields) -- and names this json.dumps as what would swallow the whole
    record. The record is still lost, as it always was; what must not happen is the loss
    escaping to a live run, or a serialisation failure being mistaken for a missing directory
    and sending the writer down the recovery path.
    """
    log = HingeDebugLog(str(tmp_path / "hinge_debug"), run_id="r")
    healed = []
    real_ensure = debuglog_module.ensure_private_dir
    monkeypatch.setattr(debuglog_module, "ensure_private_dir",
                        lambda path, **kw: (healed.append(path), real_ensure(path, **kw))[1])

    log.action("nav_chain_broken", pair_before=b"\x89PNG_raw_bytes")     # must not raise

    assert healed == []                             # not a missing directory; nothing to fix
    assert not (log.dir / "actions.jsonl").exists()  # the record is dropped, exactly as before
    log.action("the_next_record_still_lands")
    assert [r["action"] for r in _recs(log)] == ["the_next_record_still_lands"]
