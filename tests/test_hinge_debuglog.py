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
