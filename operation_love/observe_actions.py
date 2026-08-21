"""Explicit, worker-owned approvals for reviewed Observe actions.

The hub only stores and validates requests.  It never reads or drives a device:
the Worker claims a request while it is already waiting on the exact card, then
performs the action on its own thread.  This keeps the one HingeDriver instance
the sole owner of captures, item indexes, anchors, gestures and composer state.
"""
from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict


# Completed reviewed actions are retained only to make a just-retried HTTP request idempotent.
# They are not the durable audit trail (the worker/store own that), and a hub can remain open
# through arbitrarily many cards and runs.  Keep a generous recent window without turning an
# otherwise long-lived local hub into an unbounded in-memory action log.
_MAX_COMPLETED_RESULTS = 256
_MAX_PROTOCOL_STRING_LENGTH = 256


class ObserveActionBridge:
    """Thread-safe mailbox between the localhost hub and an Observe Worker.

    Requests are deliberately a tiny protocol: ``pass``, ``targeted_like_open``
    and ``send_current_suggestion``.  Coordinates, arbitrary text, and a
    generic "tap" command are not representable here.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workers = {}
        self._cards = {}
        self._pending = {}
        # Ordered by most recently completed/updated result, so eviction preserves a useful
        # replay window for delayed browser retries while bounding process memory.
        self._results: OrderedDict[str, dict] = OrderedDict()

    def register(self, worker) -> None:
        with self._lock:
            self._workers[(worker.run_id, worker.app)] = worker

    def unregister(self, worker) -> None:
        key = (worker.run_id, worker.app)
        with self._lock:
            # A delayed teardown from an old Worker must not unregister a replacement that
            # has already claimed the same run/app identity.
            if self._workers.get(key) is not worker:
                return
            self._workers.pop(key, None)
            card = self._cards.pop(key, None)
            if card and card.get("pending"):
                self._complete_locked(card["pending"], "aborted", "worker left the waiting boundary")

    def begin_card(self, worker) -> dict:
        key = (worker.run_id, worker.app)
        card = {
            "run_id": worker.run_id,
            "app": worker.app,
            "profile_token": secrets.token_urlsafe(18),
            "suggestion_token": secrets.token_urlsafe(18),
            "item": None,
            "suggestion_ready": False,
            # A command may only advance this card in order.  ``waiting`` permits a
            # pass while an opener is still being generated; ``ready`` additionally
            # permits opening the advertised item; ``sheet_open`` is the sole state
            # that permits Send.  This is kept in the hub, not inferred from a client
            # result, so a stale/replayed request cannot skip an irreversible step.
            "phase": "waiting",
            "pre_tap_published": False,
            "action": "waiting",
            "updated_at": time.time(),
            "pending": None,
        }
        with self._lock:
            old = self._cards.get(key)
            if old and old.get("pending"):
                self._complete_locked(old["pending"], "aborted",
                                      "a new card replaced the waiting boundary")
            self._cards[key] = card
        return dict(card)

    def update_suggestion(self, worker, profile_token: str, pick) -> None:
        key = (worker.run_id, worker.app)
        with self._lock:
            card = self._cards.get(key)
            if not card or card["profile_token"] != profile_token:
                return
            card["item"] = getattr(pick, "index", None) if pick is not None else None
            card["suggestion_ready"] = bool(pick is not None and getattr(pick, "text", None))
            if card["phase"] == "waiting" and card["suggestion_ready"]:
                card["phase"] = "ready"
            card["updated_at"] = time.time()

    def mark_pre_tap_published(self, worker, profile_token: str) -> None:
        """Record that the visible suggestion's release fact is durable.

        An AI action cannot race ahead of the debug fact that proves the suggestion
        was on the hub before its heart tap.  This is deliberately a Worker-side
        transition: the HTTP client cannot assert it for itself.
        """
        key = (worker.run_id, worker.app)
        with self._lock:
            card = self._cards.get(key)
            if not card or card["profile_token"] != profile_token:
                return
            card["pre_tap_published"] = True
            card["updated_at"] = time.time()

    def end_card(self, worker, profile_token: str) -> None:
        key = (worker.run_id, worker.app)
        with self._lock:
            card = self._cards.get(key)
            if card and card["profile_token"] == profile_token:
                if card.get("pending"):
                    self._complete_locked(card["pending"], "aborted", "card is no longer current")
                self._cards.pop(key, None)

    def snapshot(self, *, run_id: str | None = None, app: str | None = None) -> dict:
        with self._lock:
            cards = [
                {k: v for k, v in card.items() if k != "pending"}
                for card in self._cards.values()
                if (run_id is None or card["run_id"] == run_id)
                and (app is None or card["app"] == app)
            ]
            results = [v for v in self._results.values()
                       if (run_id is None or v["run_id"] == run_id)
                       and (app is None or v["app"] == app)]
            return {"checkpoints": cards, "results": results[-50:]}

    def submit(self, body: dict) -> tuple[bool, dict, int]:
        command = body.get("command")
        run_id, app = body.get("run_id"), body.get("app")
        profile_token, suggestion_token = body.get("profile_token"), body.get("suggestion_token")
        token = body.get("idempotency_token")
        if command not in {"pass", "targeted_like_open", "send_current_suggestion"}:
            return False, {"status": "rejected", "reason": "unsupported command"}, 400
        if not all(isinstance(x, str) and x for x in (run_id, app, profile_token, suggestion_token, token)):
            return False, {"status": "rejected", "reason": "run, app, profile, suggestion, and idempotency tokens are required"}, 400
        if any(len(value) > _MAX_PROTOCOL_STRING_LENGTH
               for value in (run_id, app, profile_token, suggestion_token, token)):
            return False, {
                "status": "rejected",
                "reason": ("run, app, profile, suggestion, and idempotency tokens must be at "
                           f"most {_MAX_PROTOCOL_STRING_LENGTH} characters"),
            }, 400
        # The public grammar is intentionally closed.  Reject rather than ignore fields that
        # could otherwise turn this reviewed protocol into arbitrary device control.
        allowed = {"command", "run_id", "app", "profile_token", "suggestion_token", "item", "idempotency_token"}
        if set(body) - allowed:
            return False, {"status": "rejected", "reason": "coordinates and arbitrary text are not accepted"}, 400
        with self._lock:
            prior = self._results.get(token) or self._pending.get(token)
            if prior:
                # An idempotency token is a replay key for one exact request, not a
                # capability that can be transplanted onto another profile/action.
                if any(prior.get(key) != body.get(key) for key in
                       ("run_id", "app", "command", "profile_token", "suggestion_token", "item")):
                    return False, {"status": "rejected",
                                   "reason": "idempotency token is bound to a different action"}, 409
                return True, dict(prior), 200
            key = (run_id, app)
            worker = self._workers.get(key)
            card = self._cards.get(key)
            if worker is None or card is None:
                return False, {"status": "rejected", "reason": "no matching worker waiting"}, 409
            if not getattr(worker, "observe_action_supported", False):
                return False, {"status": "rejected", "reason": "this worker cannot safely execute reviewed Observe actions"}, 409
            if card["profile_token"] != profile_token or card["suggestion_token"] != suggestion_token:
                return False, {"status": "rejected", "reason": "stale run/profile/suggestion binding"}, 409
            item = body.get("item")
            if command == "pass":
                if item is not None:
                    return False, {"status": "rejected", "reason": "pass does not accept an item"}, 400
                if card["phase"] not in {"waiting", "ready"}:
                    return False, {"status": "rejected",
                                   "reason": "pass is unavailable after a reviewed like action began"}, 409
            else:
                if type(item) is not int or item != card["item"] or not card["suggestion_ready"]:
                    return False, {"status": "rejected", "reason": "no current suggestion for that exact item"}, 409
                if command == "targeted_like_open":
                    if card["phase"] != "ready" or not card["pre_tap_published"]:
                        return False, {"status": "rejected",
                                       "reason": "the suggested item is not yet release-ready to open"}, 409
                elif card["phase"] != "sheet_open":
                    return False, {"status": "rejected",
                                   "reason": "send requires the matching reviewed sheet to be open"}, 409
            if card.get("pending"):
                return False, {"status": "rejected", "reason": "another reviewed action is already pending"}, 409
            result = {"idempotency_token": token, "run_id": run_id, "app": app,
                      "profile_token": profile_token, "suggestion_token": suggestion_token,
                      "command": command, "status": "queued", "item": item,
                      "updated_at": time.time()}
            self._pending[token] = result
            card["pending"] = token
            card["action"] = "queued"
            card["updated_at"] = time.time()
            return True, dict(result), 202

    def has_pending(self, worker, profile_token: str) -> bool:
        with self._lock:
            card = self._cards.get((worker.run_id, worker.app))
            return bool(card and card["profile_token"] == profile_token and card.get("pending"))

    def claim(self, worker, profile_token: str) -> dict | None:
        with self._lock:
            card = self._cards.get((worker.run_id, worker.app))
            if not card or card["profile_token"] != profile_token or not card.get("pending"):
                return None
            token = card["pending"]
            result = self._pending.get(token)
            if result is None:
                return None
            card["action"] = "executing"
            result["status"] = "executing"
            result["updated_at"] = card["updated_at"] = time.time()
            return dict(result)

    def complete(self, action: dict, status: str, reason: str | None = None,
                 *, phase: str | None = None) -> None:
        with self._lock:
            self._complete_locked(action["idempotency_token"], status, reason, phase=phase)

    def _complete_locked(self, token: str, status: str, reason: str | None = None,
                         *, phase: str | None = None) -> None:
        result = self._pending.pop(token, self._results.get(token))
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
        for card in self._cards.values():
            if card.get("pending") == token:
                card["pending"] = None
                card["action"] = status
                if phase is not None:
                    card["phase"] = phase
                elif status in {"failed", "rejected", "aborted"}:
                    # A target may have changed the visible sheet before a verifier
                    # refused it.  Never enable a different action after that ambiguity.
                    card["phase"] = "terminal"
                card["updated_at"] = time.time()
