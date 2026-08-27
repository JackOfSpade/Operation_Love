"""One-card manual training decisions for the local Hub.

The Hub is an approval surface, never a second device controller.  A Hinge
worker publishes its verified, post-keyboard-hide composer frame plus the
already-captured, top-to-bottom profile review frames, claims a human
Like/Dislike choice on its own thread, and performs the corresponding device
action itself.  The bridge owns capabilities and idempotency, not taps.
"""
from __future__ import annotations

import base64
import secrets
import struct
import threading
import time
import zlib
from collections import OrderedDict

_MAX_COMPLETED_RESULTS = 256
_MAX_PROTOCOL_STRING_LENGTH = 256
_MAX_PRE_SEND_FRAME_BYTES = 12 * 1024 * 1024
# Public because config validation must reject a Training capture budget that can never fit in a
# Hub checkpoint. Keep the protocol's one source of truth here, beside the bridge that enforces
# it at runtime.
MAX_PROFILE_REVIEW_FRAMES = 16
_MAX_PROFILE_REVIEW_BYTES = 64 * 1024 * 1024
_MAX_RASTER_DIMENSION = 10_000
_MAX_RASTER_PIXELS = 30_000_000
_MAX_PNG_CHUNKS = 10_000


def _valid_dimensions(width: int, height: int) -> bool:
    return (0 < width <= _MAX_RASTER_DIMENSION and 0 < height <= _MAX_RASTER_DIMENSION
            and width * height <= _MAX_RASTER_PIXELS)


def _valid_png_idat(*, width: int, height: int, bit_depth: int, color_type: int,
                    compression: int, filter_method: int, interlace: int,
                    data: bytes) -> bool:
    """Prove that a bounded, non-interlaced PNG has decodable pixel rows."""
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    allowed_depths = {
        0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8},
        4: {8, 16}, 6: {8, 16},
    }
    if (channels is None or bit_depth not in allowed_depths[color_type]
            or compression != 0 or filter_method != 0 or interlace != 0):
        return False
    row_bytes = (width * channels * bit_depth + 7) // 8
    expected = height * (row_bytes + 1)
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(data, expected + 1)
        return (len(raw) == expected and decoder.eof and not decoder.unconsumed_tail
                and not decoder.unused_data)
    except zlib.error:
        return False


def _raster_mime(image: object) -> str | None:
    """Recognize only a complete, bounded, browser-decodable PNG review frame.

    Android's production ``screencap -p`` output is PNG.  Reject JPEG/WebP rather than accept
    a merely structural envelope: validating their compressed payloads safely would require a
    real bounded decoder, and a broken Hub image must never leave Like/Dislike enabled.
    """
    if not isinstance(image, bytes) or not image:
        return None
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        offset = 8
        chunks = 0
        saw_idat = False
        idat = []
        png_header = None
        while offset + 12 <= len(image) and chunks < _MAX_PNG_CHUNKS:
            size = struct.unpack(">I", image[offset:offset + 4])[0]
            kind = image[offset + 4:offset + 8]
            data_start = offset + 8
            end = data_start + size + 4
            if end > len(image):
                return None
            data = image[data_start:data_start + size]
            expected_crc = struct.unpack(">I", image[data_start + size:end])[0]
            if zlib.crc32(kind + data) & 0xffffffff != expected_crc:
                return None
            if chunks == 0:
                if kind != b"IHDR" or size != 13:
                    return None
                width, height, bit_depth, color_type, compression, filter_method, interlace = (
                    struct.unpack(">IIBBBBB", data))
                if not _valid_dimensions(width, height):
                    return None
                png_header = (width, height, bit_depth, color_type, compression,
                              filter_method, interlace)
            elif kind == b"IHDR":
                return None
            if kind == b"IDAT":
                saw_idat = True
                idat.append(data)
            if kind == b"IEND":
                if size != 0 or not saw_idat or end != len(image) or png_header is None:
                    return None
                return ("image/png" if _valid_png_idat(
                    width=png_header[0], height=png_header[1], bit_depth=png_header[2],
                    color_type=png_header[3], compression=png_header[4],
                    filter_method=png_header[5], interlace=png_header[6],
                    data=b"".join(idat)) else None)
            offset, chunks = end, chunks + 1
        return None
    return None


def _image_data_url(image: object) -> str | None:
    """Encode a verified captured raster frame for the local Training Hub."""
    mime = _raster_mime(image)
    if mime is None:
        return None
    return f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"


class TrainingActionBridge:
    """Thread-safe exact-card Like/Dislike mailbox for a training Worker.

    ``submit`` only queues an action.  ``wait_for_action`` claims it and marks
    it executing; the Worker must call :meth:`complete` only after the physical
    Hinge action and its durable training record have succeeded.  This keeps a
    Hub response from claiming training data exists before it actually does.
    """

    def __init__(self) -> None:
        self._lock = threading.Condition(threading.RLock())
        self._workers = {}
        self._cards = {}
        self._pending = {}
        self._results: OrderedDict[str, dict] = OrderedDict()

    @staticmethod
    def _key(worker) -> tuple[str, str]:
        return (worker.run_id, worker.app)

    def register(self, worker) -> None:
        key = self._key(worker)
        with self._lock:
            previous = self._workers.get(key)
            if previous is not None and previous is not worker:
                # A run/app collision is never a harmless hand-off: before publishing a card,
                # a worker may still be in a multi-minute capture, navigation, or typing span
                # that touches the one physical phone.  Stop is cooperative, not immediate,
                # so no replacement may claim the mailbox (or start device work) until the
                # previous owner has actually unregistered.  Its queued card is likewise left
                # for that owner to cancel during its normal shutdown cleanup.
                stop_event = getattr(previous, "stop_event", None)
                if stop_event is not None and callable(getattr(stop_event, "set", None)):
                    stop_event.set()
                self._lock.notify_all()
                raise RuntimeError(
                    "training worker replacement refused until the prior worker unregisters")
            self._workers[key] = worker
            self._lock.notify_all()

    def unregister(self, worker) -> None:
        key = self._key(worker)
        with self._lock:
            if self._workers.get(key) is not worker:
                return
            self._workers.pop(key, None)
            card = self._cards.pop(key, None)
            if card and card.get("pending"):
                self._complete_locked(card["pending"], "aborted",
                                      "worker left training decision boundary")
            self._lock.notify_all()

    def publish_checkpoint(self, worker, pre_send_frame: bytes, pick, evidence=None,
                           profile_frames=()) -> dict:
        """Publish the immutable verified target and ordered profile review frames.

        The driver must have hidden the keyboard and then re-located/re-verified
        the composer, target, and available device controls before this call.
        This bridge deliberately does not infer that condition from a crop or
        coordinate; it only accepts the resulting complete screenshot. Supplementary profile
        frames are the worker's existing capture sequence and never authorize either action.
        """
        opener = getattr(pick, "text", None) if pick is not None else None
        if not isinstance(pre_send_frame, bytes) or not pre_send_frame:
            raise ValueError("training checkpoint requires a non-empty post-type image frame")
        if len(pre_send_frame) > _MAX_PRE_SEND_FRAME_BYTES:
            raise ValueError("training checkpoint frame exceeds the safe Hub review limit")
        image_data_url = _image_data_url(pre_send_frame)
        if image_data_url is None:
            raise ValueError("training checkpoint requires a complete PNG image frame")
        if not isinstance(opener, str) or not opener.strip():
            raise ValueError("training checkpoint requires a non-empty typed opener")
        if not isinstance(profile_frames, (list, tuple)):
            raise ValueError("training checkpoint profile frames must be an ordered sequence")
        if len(profile_frames) > MAX_PROFILE_REVIEW_FRAMES:
            raise ValueError("training checkpoint has too many profile review frames")
        review_frames = tuple(profile_frames)
        if sum(len(frame) for frame in review_frames if isinstance(frame, bytes)) \
                > _MAX_PROFILE_REVIEW_BYTES:
            raise ValueError("training checkpoint profile frames exceed the safe Hub review limit")
        for frame in review_frames:
            if (not isinstance(frame, bytes) or not frame
                    or len(frame) > _MAX_PRE_SEND_FRAME_BYTES or _raster_mime(frame) is None):
                raise ValueError(
                    "training checkpoint profile review requires complete PNG frames")
        key = self._key(worker)
        card = {
            "run_id": worker.run_id,
            "app": worker.app,
            "profile_token": secrets.token_urlsafe(18),
            "approval_token": secrets.token_urlsafe(18),
            "phase": "waiting_training_decision",
            "opener": opener,
            "referenced": getattr(pick, "referenced", None) if pick is not None else None,
            "item": getattr(pick, "index", None) if pick is not None else None,
            "item_description": getattr(pick, "item_description", None) if pick is not None else None,
            "image_data_url": image_data_url,
            # Kept as bytes behind a separate, token-bound image endpoint. Embedding every
            # capture in the one-second checkpoint JSON poll would resend tens of megabytes
            # even while the reviewer was looking at only one frame.
            "profile_image_count": len(review_frames),
            "_profile_frames": review_frames,
            "evidence_id": ((evidence or {}).get("evidence_id")
                            if isinstance(evidence, dict) else None),
            "action": "ready",
            "pending": None,
            "updated_at": time.time(),
        }
        with self._lock:
            if self._workers.get(key) is not worker:
                raise ValueError("training worker is not registered with the action bridge")
            if not getattr(worker, "training_action_supported", False):
                raise ValueError("worker is not authorized for training decisions")
            stop_event = getattr(worker, "stop_event", None)
            if stop_event is not None and stop_event.is_set():
                raise ValueError("training run is stopping; checkpoint was not published")
            old = self._cards.get(key)
            if old and old.get("pending"):
                self._complete_locked(old["pending"], "aborted",
                                      "a new checkpoint replaced the training decision")
            self._cards[key] = card
            self._lock.notify_all()
            return {name: value for name, value in card.items()
                    if not name.startswith("_")}

    # A descriptive alias keeps the bridge easy to use from a driver callback without
    # preserving AUTO-testing terminology in the public protocol.
    begin_checkpoint = publish_checkpoint

    def profile_review_image(self, *, run_id: str, app: str,
                             profile_token: str, index: int) -> bytes | None:
        """Return one immutable top-to-bottom capture for the exact live checkpoint."""
        if (not all(isinstance(value, str) and value for value in
                    (run_id, app, profile_token))
                or isinstance(index, bool) or not isinstance(index, int)):
            return None
        with self._lock:
            card = self._cards.get((run_id, app))
            if card is None or card.get("profile_token") != profile_token:
                return None
            frames = card.get("_profile_frames", ())
            if not 0 <= index < len(frames):
                return None
            return frames[index]

    def wait_for_action(self, worker, profile_token: str,
                        stop_event: threading.Event) -> dict | None:
        """Claim the one current action, or return ``None`` without releasing anything.

        Claiming is not completion.  The returned action remains represented as
        ``executing`` until the Worker reports the real device/storage outcome.
        """
        key = self._key(worker)
        with self._lock:
            while not stop_event.is_set():
                if self._workers.get(key) is not worker:
                    return None
                card = self._cards.get(key)
                if card is None or card.get("profile_token") != profile_token:
                    return None
                token = card.get("pending")
                if token:
                    action = self._pending.get(token)
                    if action is not None and action.get("status") == "queued":
                        card["action"] = "executing"
                        card["phase"] = "executing_training_decision"
                        card["updated_at"] = time.time()
                        action["status"] = "executing"
                        action["updated_at"] = card["updated_at"]
                        return dict(action)
                self._lock.wait(timeout=0.25)
            return None

    def complete(self, action: dict, status: str, reason: str | None = None) -> bool:
        """Finish a claimed action after its real worker-owned outcome."""
        if status not in {"completed", "failed", "aborted"}:
            raise ValueError("training action status must be completed, failed, or aborted")
        token = action.get("idempotency_token") if isinstance(action, dict) else None
        if not isinstance(token, str) or not token:
            raise ValueError("training action requires its idempotency token")
        with self._lock:
            # Completion is intentionally narrower than cancellation: only the exact action
            # still executing in the live mailbox may become a terminal result.  This makes a
            # result immutable after replacement/unregister and prevents a late old worker
            # from rewriting an aborted outcome as completed.
            pending = self._pending.get(token)
            if pending is None or pending.get("status") != "executing":
                return False
            if any(pending.get(name) != action.get(name) for name in (
                    "run_id", "app", "command", "profile_token", "approval_token")):
                return False
            self._complete_locked(token, status, reason)
            self._lock.notify_all()
            return True

    def cancel_checkpoint(self, worker, profile_token: str,
                          reason: str = "run is stopping") -> None:
        key = self._key(worker)
        with self._lock:
            card = self._cards.get(key)
            if card is not None and card.get("profile_token") == profile_token:
                if card.get("pending"):
                    self._complete_locked(card["pending"], "aborted", reason)
                self._cards.pop(key, None)
            self._lock.notify_all()

    def snapshot(self, *, run_id: str | None = None, app: str | None = None) -> dict:
        with self._lock:
            cards = []
            for card in self._cards.values():
                if ((run_id is not None and card["run_id"] != run_id)
                        or (app is not None and card["app"] != app)):
                    continue
                payload = {name: value for name, value in card.items()
                           if name != "pending" and not name.startswith("_")}
                payload["pending"] = (card["phase"] == "waiting_training_decision"
                                      and card["action"] == "ready")
                token = card.get("pending")
                action = self._pending.get(token) if token else None
                if action is not None:
                    # This is status-only feedback for the exact checkpoint already visible in
                    # the Hub, not a capability: the approval token remains required to submit.
                    payload["command"] = action.get("command")
                cards.append(payload)
            results = [dict(result) for result in self._results.values()
                       if (run_id is None or result["run_id"] == run_id)
                       and (app is None or result["app"] == app)]
            return {"checkpoints": cards, "results": results[-50:]}

    def submit(self, body: dict) -> tuple[bool, dict, int]:
        if not isinstance(body, dict):
            return False, {"status": "rejected", "reason": "action body must be an object"}, 400
        command = body.get("command")
        run_id, app = body.get("run_id"), body.get("app")
        profile_token, approval_token = body.get("profile_token"), body.get("approval_token")
        token = body.get("idempotency_token")
        if not isinstance(command, str) or command not in {"like", "dislike"}:
            return False, {"status": "rejected", "reason": "unsupported command"}, 400
        required = (run_id, app, profile_token, approval_token, token)
        if not all(isinstance(value, str) and value for value in required):
            return False, {"status": "rejected", "reason": (
                "run, app, profile, approval, and idempotency tokens are required")}, 400
        if any(len(value) > _MAX_PROTOCOL_STRING_LENGTH for value in required):
            return False, {"status": "rejected", "reason": "protocol tokens are too long"}, 400
        allowed = {"command", "run_id", "app", "profile_token", "approval_token",
                   "idempotency_token"}
        if set(body) - allowed:
            return False, {"status": "rejected", "reason": "unexpected action fields"}, 400
        with self._lock:
            prior = self._results.get(token) or self._pending.get(token)
            if prior:
                if any(prior.get(name) != body.get(name) for name in
                       ("run_id", "app", "command", "profile_token", "approval_token")):
                    return False, {"status": "rejected", "reason": (
                        "idempotency token is bound to a different action")}, 409
                # A replay is idempotent, but a previously failed/aborted operation is not a
                # successful approval.  Preserve the terminal result and make the HTTP wrapper
                # truthful so the UI can surface the reason instead of silently hiding it.
                if prior.get("status") in {"failed", "aborted"}:
                    return False, dict(prior), 409
                return True, dict(prior), 200
            key = (run_id, app)
            worker = self._workers.get(key)
            card = self._cards.get(key)
            if worker is None or card is None:
                return False, {"status": "rejected", "reason": "no matching worker waiting"}, 409
            if not getattr(worker, "training_action_supported", False):
                return False, {"status": "rejected", "reason": (
                    "this worker cannot execute training decisions")}, 409
            stop_event = getattr(worker, "stop_event", None)
            if stop_event is not None and stop_event.is_set():
                return False, {"status": "rejected", "reason": (
                    "run is stopping; training decision is no longer available")}, 409
            if (card["profile_token"] != profile_token
                    or card["approval_token"] != approval_token):
                return False, {"status": "rejected", "reason": (
                    "stale run/profile/approval binding")}, 409
            if card.get("pending"):
                return False, {"status": "rejected", "reason": (
                    "a training decision is already pending")}, 409
            result = {"idempotency_token": token, "run_id": run_id, "app": app,
                      "profile_token": profile_token, "approval_token": approval_token,
                      "command": command, "status": "queued", "updated_at": time.time()}
            self._pending[token] = result
            card["pending"] = token
            card["action"] = "queued"
            card["updated_at"] = time.time()
            self._lock.notify_all()
            return True, dict(result), 202

    def _complete_locked(self, token: str, status: str, reason: str | None = None) -> None:
        # Results are terminal.  Internal cancellation/replacement paths call this helper too,
        # so do not let any late path pull an already completed result back out of _results and
        # mutate it.
        result = self._pending.pop(token, None)
        if result is None:
            return
        result = dict(result)
        result["status"] = status
        result["updated_at"] = time.time()
        if reason:
            result["reason"] = reason
        self._results[token] = result
        self._results.move_to_end(token)
        while len(self._results) > _MAX_COMPLETED_RESULTS:
            self._results.popitem(last=False)
        for key, card in tuple(self._cards.items()):
            if card.get("pending") == token:
                self._cards.pop(key, None)
