"""Training Hub mailbox: exact-card Like/Dislike, cancellation, and replay."""
from __future__ import annotations

import struct
import threading
import zlib

import pytest

from operation_love.training_actions import TrainingActionBridge


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))


_FRAME = (b"\x89PNG\r\n\x1a\n"
          + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
          + _chunk(b"IDAT", zlib.compress(b"\x00\x00")) + _chunk(b"IEND", b""))


class _Worker:
    run_id = "training-run"
    app = "hinge"
    training_action_supported = True

    def __init__(self):
        self.stop_event = threading.Event()


class _Pick:
    text = "The typed opener"
    referenced = "mountain photo"
    index = 2
    item_description = "mountain photo"


def _body(card, command="like", token="request-1"):
    return {"command": command, "run_id": card["run_id"], "app": card["app"],
            "profile_token": card["profile_token"], "approval_token": card["approval_token"],
            "idempotency_token": token}


def test_training_checkpoint_has_only_valid_like_dislike_capability():
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    card = bridge.publish_checkpoint(worker, _FRAME, _Pick(), {"evidence_id": "post-hide"})
    checkpoint = bridge.snapshot()["checkpoints"][0]
    assert checkpoint["phase"] == "waiting_training_decision"
    assert checkpoint["pending"] is True
    assert checkpoint["opener"] == _Pick.text
    assert checkpoint["evidence_id"] == "post-hide"
    assert checkpoint["image_data_url"].startswith("data:image/png;base64,")
    assert bridge.submit(_body(card, "continue"))[2] == 400


def test_training_checkpoint_rejects_oversized_frame_without_publishing_card():
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)

    with pytest.raises(ValueError, match="safe Hub review limit"):
        bridge.publish_checkpoint(worker, b"x" * (12 * 1024 * 1024 + 1), _Pick())

    assert bridge.snapshot()["checkpoints"] == []


@pytest.mark.parametrize(("frame", "text", "match"), [
    (b"", "The typed opener", "post-type image"),
    (b"not a raster", "The typed opener", "complete PNG"),
    (b"\x89PNG\r\n\x1a\n", "The typed opener", "complete PNG"),
    (b"\xff\xd8not a complete JPEG\xff\xd9", "The typed opener", "complete PNG"),
    (_FRAME, " \n\t ", "typed opener"),
])
def test_training_checkpoint_never_publishes_without_complete_review_data(frame, text, match):
    bridge, worker = TrainingActionBridge(), _Worker()
    worker_pick = type("Pick", (), {"text": text})()
    bridge.register(worker)

    with pytest.raises(ValueError, match=match):
        bridge.publish_checkpoint(worker, frame, worker_pick)

    assert bridge.snapshot()["checkpoints"] == []


def test_training_checkpoint_rejects_crc_valid_png_with_undecodable_pixel_data():
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    malformed = (b"\x89PNG\r\n\x1a\n"
                 + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
                 + _chunk(b"IDAT", b"not a zlib stream")
                 + _chunk(b"IEND", b""))

    with pytest.raises(ValueError, match="complete PNG"):
        bridge.publish_checkpoint(worker, malformed, _Pick())

    assert bridge.snapshot()["checkpoints"] == []


def test_training_checkpoint_refuses_unregistered_worker_instead_of_stranding_card():
    bridge = TrainingActionBridge()

    with pytest.raises(ValueError, match="not registered"):
        bridge.publish_checkpoint(_Worker(), _FRAME, _Pick())

    assert bridge.snapshot()["checkpoints"] == []


def test_like_claims_then_only_completion_retires_training_checkpoint():
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    card = bridge.publish_checkpoint(worker, _FRAME, _Pick())
    ok, queued, code = bridge.submit(_body(card, "like"))
    assert (ok, queued["status"], code) == (True, "queued", 202)
    action = bridge.wait_for_action(worker, card["profile_token"], worker.stop_event)
    assert action and action["command"] == "like" and action["status"] == "executing"
    live = bridge.snapshot()["checkpoints"][0]
    assert live["pending"] is False and live["action"] == "executing"
    bridge.complete(action, "completed")
    assert bridge.snapshot()["checkpoints"] == []
    assert bridge.submit(_body(card, "like"))[1]["status"] == "completed"


def test_dislike_replay_is_idempotent_and_bound_to_the_exact_card():
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    first = bridge.publish_checkpoint(worker, _FRAME, _Pick())
    ok, result, code = bridge.submit(_body(first, "dislike"))
    assert (ok, code, result["status"]) == (True, 202, "queued")
    assert bridge.submit(_body(first, "dislike"))[2] == 200
    assert bridge.submit(_body(first, "like"))[2] == 409
    second = bridge.publish_checkpoint(worker, _FRAME, _Pick())
    assert bridge.submit(_body(first, "dislike", "stale"))[2] == 409
    bridge.cancel_checkpoint(worker, second["profile_token"])


def test_stop_or_worker_replacement_never_claims_an_action():
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    card = bridge.publish_checkpoint(worker, _FRAME, _Pick())
    worker.stop_event.set()
    assert bridge.wait_for_action(worker, card["profile_token"], worker.stop_event) is None
    bridge.cancel_checkpoint(worker, card["profile_token"])
    bridge.unregister(worker)

    old, replacement = _Worker(), _Worker()
    bridge.register(old)
    old_card = bridge.publish_checkpoint(old, _FRAME, _Pick())
    with pytest.raises(RuntimeError, match="prior worker unregisters"):
        bridge.register(replacement)
    assert old.stop_event.is_set()
    assert bridge.wait_for_action(old, old_card["profile_token"], old.stop_event) is None
    bridge.unregister(old)
    bridge.register(replacement)


def test_replacement_stops_displaced_worker_and_refuses_until_cleanup():
    """Two same-run workers must never race one phone at any device boundary."""
    bridge, old, replacement = TrainingActionBridge(), _Worker(), _Worker()
    bridge.register(old)
    card = bridge.publish_checkpoint(old, _FRAME, _Pick())
    assert bridge.submit(_body(card, "like"))[2] == 202
    action = bridge.wait_for_action(old, card["profile_token"], old.stop_event)
    assert action and action["status"] == "executing"

    with pytest.raises(RuntimeError, match="prior worker unregisters"):
        bridge.register(replacement)
    assert old.stop_event.is_set()
    assert bridge.snapshot()["checkpoints"][0]["action"] == "executing"

    # Once the old worker leaves, its already-aborted result is immutable: a late completion
    # must not turn a displaced device action into a successful training record.
    bridge.unregister(old)
    assert bridge.snapshot()["results"][-1]["status"] == "aborted"
    assert bridge.complete(action, "completed") is False
    assert bridge.snapshot()["results"][-1]["status"] == "aborted"
    bridge.register(replacement)


def test_replacement_waits_for_prior_cleanup_before_new_owner_registers():
    bridge, old, replacement = TrainingActionBridge(), _Worker(), _Worker()
    bridge.register(old)
    card = bridge.publish_checkpoint(old, _FRAME, _Pick())
    assert bridge.submit(_body(card, "dislike"))[2] == 202

    with pytest.raises(RuntimeError, match="prior worker unregisters"):
        bridge.register(replacement)

    assert old.stop_event.is_set()
    assert bridge.snapshot()["checkpoints"][0]["action"] == "queued"
    bridge.unregister(old)
    assert bridge.snapshot()["checkpoints"] == []
    assert bridge.snapshot()["results"][-1]["status"] == "aborted"
    bridge.register(replacement)


def test_replacement_without_a_card_still_waits_for_prior_worker_to_exit():
    bridge, old, replacement = TrainingActionBridge(), _Worker(), _Worker()
    bridge.register(old)

    with pytest.raises(RuntimeError, match="prior worker unregisters"):
        bridge.register(replacement)

    assert old.stop_event.is_set()
    bridge.unregister(old)
    bridge.register(replacement)


def test_only_executing_action_can_complete_and_failed_replay_is_not_success():
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    card = bridge.publish_checkpoint(worker, _FRAME, _Pick())
    ok, queued, code = bridge.submit(_body(card, "like"))
    assert (ok, code, queued["status"]) == (True, 202, "queued")
    assert bridge.complete(queued, "completed") is False
    assert bridge.snapshot()["checkpoints"][0]["action"] == "queued"

    action = bridge.wait_for_action(worker, card["profile_token"], worker.stop_event)
    assert action
    assert bridge.complete(action, "failed", "storage unavailable") is True
    ok, replay, code = bridge.submit(_body(card, "like"))
    assert (ok, code, replay["status"], replay["reason"]) == (
        False, 409, "failed", "storage unavailable")
