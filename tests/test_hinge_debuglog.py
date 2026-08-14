"""Host-side debug log for the Hinge driver — action records + rotating screenshots, with
error shots kept forever (so a failure that halts the run is never rotated away)."""
import json

from operation_love.drivers.debuglog import HingeDebugLog


def _recs(d):
    return [json.loads(ln) for ln in (d.dir / "actions.jsonl").read_text().splitlines()]


def test_action_writes_record_and_before_after_shots(tmp_path):
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.action("like", before=b"AAA", after=b"BBB", heart=[937, 1600], rose_modal=True)
    rec = _recs(dl)[0]
    assert rec["action"] == "like" and rec["heart"] == [937, 1600] and rec["rose_modal"] is True
    assert (dl.dir / rec["before"]).read_bytes() == b"AAA"
    assert (dl.dir / rec["after"]).read_bytes() == b"BBB"


def test_action_can_preserve_an_identity_anchor_shot(tmp_path):
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.action("capture_split", before=b"TOP", anchor=b"ANCHOR", after=b"TRIGGER")

    rec = _recs(dl)[0]
    assert (dl.dir / rec["anchor"]).read_bytes() == b"ANCHOR"


def test_normal_shots_rotate_but_error_shots_are_kept(tmp_path):
    dl = HingeDebugLog(str(tmp_path), run_id="r", keep_shots=4)
    dl.error("unexpected", b"CRITICAL", ValueError("boom"))   # error shot — must survive rotation
    for i in range(20):                                        # flood with normal shots
        dl.action("dislike", before=bytes([i]), after=bytes([i + 64]))
    pngs = list(dl.dir.glob("*.png"))
    assert len(pngs) <= 4 + 1                                  # rotating cap + the kept error shot
    err = _recs(dl)[0]
    assert err["action"] == "unexpected" and "boom" in err["error"]
    assert (dl.dir / err["screenshot"]).read_bytes() == b"CRITICAL"   # crucial log preserved


def test_logging_is_best_effort_no_frame(tmp_path):
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.action("capture", before=None, photos=0)               # no screenshot -> no shot keys
    rec = _recs(dl)[0]
    assert rec["action"] == "capture" and "before" not in rec


def test_screenshot_write_failure_is_swallowed_but_record_still_written(tmp_path, monkeypatch):
    import pathlib
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    def boom(self, data):                                      # disk dies mid-write
        raise OSError("disk full")
    monkeypatch.setattr(pathlib.Path, "write_bytes", boom)
    dl.action("like", before=b"PNGDATA")                       # must not raise despite the bad write
    dl.error("unexpected", b"PNGDATA", RuntimeError("x"))      # error shots are best-effort too
    recs = _recs(dl)
    assert recs[0]["action"] == "like" and "before" not in recs[0]   # line written, just no shot key
    assert recs[1]["action"] == "unexpected" and "screenshot" not in recs[1]


def test_jsonl_append_failure_is_swallowed(tmp_path, monkeypatch):
    import pathlib
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    real_open = pathlib.Path.open
    def boom(self, *a, **k):                                   # only the actions.jsonl append fails
        if self.name == "actions.jsonl":
            raise OSError("disk full")
        return real_open(self, *a, **k)
    monkeypatch.setattr(pathlib.Path, "open", boom)
    dl.action("dislike")                                       # must not raise despite the bad append


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
    """rotate=False (error) shots must keep behaving exactly as before dedup existed: always a
    fresh write, kept forever, and -- per the scope documented in __init__ -- never consulted
    as a dedup source for normal shots either, so this path stays as unconditional as it was."""
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.error("unexpected", b"SAME_BYTES", ValueError("boom"))
    dl.action("dislike", before=b"SAME_BYTES")                 # byte-identical to the error shot

    recs = _recs(dl)
    assert recs[0]["screenshot"] != recs[1]["before"]          # each got its own file
    assert len(list(dl.dir.glob("*.png"))) == 2


def test_two_identical_error_shots_both_write_their_own_file(tmp_path):
    """Error shots were already exempt from rotation ("kept forever"); dedup must not change
    that either -- two error shots with identical bytes still both land on disk."""
    dl = HingeDebugLog(str(tmp_path), run_id="r")
    dl.error("unexpected", b"CRASH_FRAME", ValueError("boom1"))
    dl.error("unexpected", b"CRASH_FRAME", ValueError("boom2"))

    recs = _recs(dl)
    assert recs[0]["screenshot"] != recs[1]["screenshot"]
    assert len(list(dl.dir.glob("*.png"))) == 2


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
