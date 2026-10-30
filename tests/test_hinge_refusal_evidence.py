"""Every retained failing transition needs both original frames for offline replay."""

import json
from types import SimpleNamespace

from operation_love.drivers import frameshift, hinge, segment
from operation_love.drivers.debuglog import HingeDebugLog


def _driver(tmp_path):
    driver = hinge.HingeDriver.__new__(hinge.HingeDriver)
    driver._dbg = HingeDebugLog(str(tmp_path), run_id="refused-pair-priority")
    return driver


def _index(count, failed_pairs=()):
    return SimpleNamespace(
        frames=tuple(
            segment.FrameSegmentation(
                frame_size=(1080, 2400),
                frame_digest=f"frame-{i}",
                band=(300, 2100),
                card_x=(53, 1027),
                blocks=(),
                runs=(),
                hearts=(),
                unassigned_hearts=(),
                failures=(),
            )
            for i in range(count)
        ),
        source_frame_indices=tuple(range(count)),
        offsets=(0,) * count,
        shifts=tuple(
            frameshift.ShiftEstimate(
                delta_px=None if i in failed_pairs else 400,
                consensus_px=None if i in failed_pairs else 400,
                confidence=0.0,
                saturated=False,
                reason="synthetic shift ledger",
                strips=(),
                status="no_consensus" if i in failed_pairs else "measured",
                frame_size=(1080, 2400),
                band=(300, 2100),
                trust_window_px=900,
                agreeing=0,
                dissenting=0,
                eligible=0,
            )
            for i in range(count - 1)
        ),
        failures=(),
    )


def test_later_refused_pair_keeps_both_frames_before_neighbours(tmp_path):
    driver = _driver(tmp_path)
    photos = [f"frame-{i}".encode() for i in range(19)]
    index = _index(len(photos), failed_pairs=(7, 14))

    driver._item_index_refused(photos, "frames 7 and 8 have no consensus", index)

    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    refusal = records[-1]
    payload = json.loads((driver._dbg.dir / refusal["evidence_sidecar"]).read_text())
    retained = {frame["local_frame_index"] for frame in payload["frames"]}
    assert {7, 8, 14, 15} <= retained
    assert len(retained) == len(refusal["evidence_frames"]) == 8
    for frame in payload["frames"]:
        assert (driver._dbg.dir / frame["screenshot"]).read_bytes() == photos[
            frame["local_frame_index"]
        ]


def test_oversized_priority_list_cannot_exceed_the_eight_image_cap(tmp_path):
    driver = _driver(tmp_path)
    photos = [f"frame-{i}".encode() for i in range(15)]

    sidecar, saved = driver._save_item_index_refusal_evidence(
        photos,
        "many failed frames",
        _index(len(photos)),
        list(range(len(photos))),
        {},
        priority_local=tuple(range(len(photos))),
    )

    payload = json.loads((driver._dbg.dir / sidecar).read_text())
    assert len(saved) == len(payload["frames"]) == 8
    assert len(payload["all_frame_geometry"]) == len(photos)


def test_measured_contradictions_keep_their_named_pairs_with_raw_refusals(tmp_path):
    driver = _driver(tmp_path)
    photos = [f"frame-{i}".encode() for i in range(25)]
    index = _index(len(photos), failed_pairs=(14, 20))
    index.failures = (
        "frames 1 and 2 have contradictory page motion",
        "frames 8 and 9 have contradictory page motion",
    )

    driver._item_index_refused(photos, "frames 14 and 15 have no consensus", index)

    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    payload = json.loads(
        (driver._dbg.dir / records[-1]["evidence_sidecar"]).read_text()
    )
    assert {frame["local_frame_index"] for frame in payload["frames"]} == {
        1,
        2,
        8,
        9,
        14,
        15,
        20,
        21,
    }


def test_refusal_thumbnails_follow_the_same_source_mapping_as_the_dossier(tmp_path):
    driver = _driver(tmp_path)
    photos = [f"frame-{i}".encode() for i in range(8)]
    index = _index(4, failed_pairs=(1,))
    index.source_frame_indices = (0, 2, 5, 7)

    driver._item_index_refused(photos, "frames 1 and 2 have no consensus", index)

    records = [
        json.loads(line)
        for line in (driver._dbg.dir / "actions.jsonl").read_text().splitlines()
    ]
    refusal = records[-1]
    assert refusal["failing_pair"] == [1, 2]
    assert (driver._dbg.dir / refusal["before"]).read_bytes() == photos[2]
    assert (driver._dbg.dir / refusal["after"]).read_bytes() == photos[5]
