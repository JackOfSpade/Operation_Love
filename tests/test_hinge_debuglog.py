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
