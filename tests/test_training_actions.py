"""Training Hub mailbox: exact-card Like/Dislike, cancellation, and replay."""
from __future__ import annotations

import struct
import threading
import zlib
from unittest.mock import patch

import pytest

from operation_love.training_actions import TrainingActionBridge, _valid_png_idat


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


def test_training_checkpoint_exposes_ordered_profile_images_only_through_bound_endpoint():
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    card = bridge.publish_checkpoint(
        worker, _FRAME, _Pick(), profile_frames=(_FRAME, _FRAME))

    checkpoint = bridge.snapshot()["checkpoints"][0]
    assert checkpoint["profile_image_count"] == 2
    assert "_profile_frames" not in checkpoint
    assert "_profile_frames" not in card
    assert bridge.profile_review_image(
        run_id=card["run_id"], app=card["app"],
        profile_token=card["profile_token"], index=0) == _FRAME
    assert bridge.profile_review_image(
        run_id=card["run_id"], app=card["app"],
        profile_token=card["profile_token"], index=1) == _FRAME
    assert bridge.profile_review_image(
        run_id=card["run_id"], app=card["app"],
        profile_token="stale", index=0) is None
    assert bridge.profile_review_image(
        run_id=card["run_id"], app=card["app"],
        profile_token=card["profile_token"], index=2) is None


def test_training_checkpoint_rejects_malformed_supplementary_profile_frame():
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)

    with pytest.raises(ValueError, match="complete PNG frames"):
        bridge.publish_checkpoint(
            worker, _FRAME, _Pick(), profile_frames=(_FRAME, b"not png"))

    assert bridge.snapshot()["checkpoints"] == []


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


def test_training_checkpoint_rejects_a_decompressed_png_with_an_invalid_row_filter():
    """A valid CRC and zlib stream are not enough: filter byte 5 is outside PNG's 0..4 range.

    The compact 1x1 payload reaches the prior size-only validator but every browser rejects it
    while reconstructing scanlines. It must never publish a checkpoint whose Hub review image
    cannot actually render.
    """
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    malformed = (b"\x89PNG\r\n\x1a\n"
                 + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
                 + _chunk(b"IDAT", zlib.compress(b"\x05\x00")) + _chunk(b"IEND", b""))

    with pytest.raises(ValueError, match="complete PNG"):
        bridge.publish_checkpoint(worker, malformed, _Pick())

    assert bridge.snapshot()["checkpoints"] == []


def test_png_validator_rejects_large_inflated_rasters_before_decompression():
    """A compact zlib bomb must not allocate the 240 MB permitted by pixel dimensions alone."""
    with patch("operation_love.training_actions.zlib.decompressobj",
               side_effect=AssertionError("large raster must not be decompressed")):
        assert not _valid_png_idat(
            width=3_000, height=3_000, bit_depth=16, color_type=6,
            compression=0, filter_method=0, interlace=0, data=b"tiny")


def test_training_checkpoint_requires_a_palette_for_indexed_png_frames():
    """Colour type 3 needs PLTE before IDAT; zlib rows alone cannot render it."""
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    header = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 3, 0, 0, 0))
    data = _chunk(b"IDAT", zlib.compress(b"\x00\x00"))
    malformed = b"\x89PNG\r\n\x1a\n" + header + data + _chunk(b"IEND", b"")

    with pytest.raises(ValueError, match="complete PNG"):
        bridge.publish_checkpoint(worker, malformed, _Pick())

    complete = (b"\x89PNG\r\n\x1a\n" + header + _chunk(b"PLTE", b"\x00\x00\x00")
                + data + _chunk(b"IEND", b""))
    checkpoint = bridge.publish_checkpoint(worker, complete, _Pick())
    assert checkpoint["image_data_url"].startswith("data:image/png;base64,")


def test_training_checkpoint_rejects_palette_on_grayscale_png_frames():
    """PLTE is forbidden for grayscale types 0/4 even when its bytes and IDAT are valid."""
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    malformed = (b"\x89PNG\r\n\x1a\n"
                 + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
                 + _chunk(b"PLTE", b"\x00\x00\x00")
                 + _chunk(b"IDAT", zlib.compress(b"\x00\x00")) + _chunk(b"IEND", b""))

    with pytest.raises(ValueError, match="complete PNG"):
        bridge.publish_checkpoint(worker, malformed, _Pick())


def test_training_checkpoint_rejects_unknown_critical_png_chunks():
    """Uppercase chunk names are critical: a browser cannot ignore an unknown one."""
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    malformed = (b"\x89PNG\r\n\x1a\n"
                 + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
                 + _chunk(b"ABCD", b"")
                 + _chunk(b"IDAT", zlib.compress(b"\x00\x00")) + _chunk(b"IEND", b""))

    with pytest.raises(ValueError, match="complete PNG"):
        bridge.publish_checkpoint(worker, malformed, _Pick())


@pytest.mark.parametrize("chunk_type", [b"a1cd", b"abcd"])
def test_training_checkpoint_rejects_malformed_png_chunk_types(chunk_type):
    """Chunk type bytes must be ASCII letters and retain PNG's uppercase reserved bit."""
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    malformed = (b"\x89PNG\r\n\x1a\n"
                 + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
                 + _chunk(chunk_type, b"")
                 + _chunk(b"IDAT", zlib.compress(b"\x00\x00")) + _chunk(b"IEND", b""))

    with pytest.raises(ValueError, match="complete PNG"):
        bridge.publish_checkpoint(worker, malformed, _Pick())


def test_training_checkpoint_rejects_a_truecolor_palette_with_over_256_entries():
    """Truecolour's 16-bit depth does not expand PNG PLTE beyond its fixed 256 entries."""
    bridge, worker = TrainingActionBridge(), _Worker()
    bridge.register(worker)
    malformed = (b"\x89PNG\r\n\x1a\n"
                 + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0))
                 + _chunk(b"PLTE", b"\x00\x00\x00" * 257)
                 + _chunk(b"IDAT", zlib.compress(b"\x00" + b"\x00" * 6))
                 + _chunk(b"IEND", b""))

    with pytest.raises(ValueError, match="complete PNG"):
        bridge.publish_checkpoint(worker, malformed, _Pick())


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
