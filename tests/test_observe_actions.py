"""Offline contract tests for the reviewed Observe action mailbox (no ADB/network)."""
import threading
from types import SimpleNamespace

import pytest

from operation_love.drivers.base import ItemTargetingError
from operation_love.observe_actions import ObserveActionBridge
from operation_love.worker import Worker, _OBSERVE_PRE_TAP_TARGETING_REFUSED


class _Worker:
    run_id = "run-1"
    app = "hinge"
    observe_action_supported = True


def _bound_card():
    bridge, worker = ObserveActionBridge(), _Worker()
    bridge.register(worker)
    card = bridge.begin_card(worker)
    bridge.update_suggestion(worker, card["profile_token"], SimpleNamespace(index=3, text="hello"))
    bridge.mark_pre_tap_published(worker, card["profile_token"])
    return bridge, worker, card


def _request(card, command="targeted_like_open", **extra):
    body = {"command": command, "run_id": card["run_id"], "app": card["app"],
            "profile_token": card["profile_token"], "suggestion_token": card["suggestion_token"],
            "idempotency_token": "review-1"}
    if command != "pass":
        body["item"] = 3
    body.update(extra)
    return body


def test_reviewed_action_requires_exact_run_card_suggestion_and_item():
    bridge, _worker, card = _bound_card()
    for body in (_request(card, run_id="old-run"), _request(card, profile_token="old-card"),
                 _request(card, suggestion_token="old-suggestion"), _request(card, item=2)):
        ok, result, code = bridge.submit(body)
        assert not ok and code == 409 and result["status"] == "rejected"


def test_no_suggestion_coordinates_or_arbitrary_text_are_rejected():
    bridge, worker, card = ObserveActionBridge(), _Worker(), None
    bridge.register(worker)
    card = bridge.begin_card(worker)
    ok, result, code = bridge.submit(_request(card))
    assert not ok and code == 409 and "suggestion" in result["reason"]
    bridge.update_suggestion(worker, card["profile_token"], SimpleNamespace(index=3, text="hello"))
    ok, result, code = bridge.submit(_request(card, x=10))
    assert not ok and code == 400 and "coordinates" in result["reason"]


def test_idempotency_and_one_pending_command_make_race_safe():
    bridge, worker, card = _bound_card()
    ok, first, code = bridge.submit(_request(card))
    assert ok and code == 202 and first["status"] == "queued"
    # A replay returns the original result; a distinct token cannot race a second device action.
    assert bridge.submit(_request(card))[1]["status"] == "queued"
    ok, result, code = bridge.submit(_request(card, idempotency_token="review-2"))
    assert not ok and code == 409 and "already pending" in result["reason"]
    claimed = bridge.claim(worker, card["profile_token"])
    bridge.complete(claimed, "completed")
    assert bridge.submit(_request(card))[1]["status"] == "completed"


def test_send_requires_successful_open_and_pass_is_blocked_once_opened():
    bridge, worker, card = _bound_card()
    ok, result, code = bridge.submit(_request(card, command="send_current_suggestion"))
    assert not ok and code == 409 and "sheet" in result["reason"]
    assert bridge.submit(_request(card))[0]
    opened = bridge.claim(worker, card["profile_token"])
    bridge.complete(opened, "completed", phase="sheet_open")
    ok, result, code = bridge.submit(_request(card, command="pass", idempotency_token="pass-after-open"))
    assert not ok and code == 409 and "after" in result["reason"]
    ok, queued, code = bridge.submit(_request(card, command="send_current_suggestion",
                                               idempotency_token="send-after-open"))
    assert ok and code == 202 and queued["status"] == "queued"
    sent = bridge.claim(worker, card["profile_token"])
    bridge.complete(sent, "completed", phase="terminal")
    ok, result, code = bridge.submit(_request(card, command="send_current_suggestion",
                                               idempotency_token="send-again"))
    assert not ok and code == 409 and "sheet" in result["reason"]


def test_open_requires_durable_pre_tap_publication_and_failures_are_terminal():
    bridge, worker = ObserveActionBridge(), _Worker()
    bridge.register(worker)
    card = bridge.begin_card(worker)
    bridge.update_suggestion(worker, card["profile_token"], SimpleNamespace(index=3, text="hello"))
    ok, result, code = bridge.submit(_request(card))
    assert not ok and code == 409 and "release-ready" in result["reason"]
    bridge.mark_pre_tap_published(worker, card["profile_token"])
    assert bridge.submit(_request(card))[0]
    claimed = bridge.claim(worker, card["profile_token"])
    bridge.complete(claimed, "failed", "verification refused")
    ok, result, code = bridge.submit(_request(card, command="pass", idempotency_token="after-failure"))
    assert not ok and code == 409 and "after" in result["reason"]


def test_idempotency_token_cannot_replay_a_different_card_or_command():
    bridge, _worker, card = _bound_card()
    assert bridge.submit(_request(card))[0]
    ok, result, code = bridge.submit(_request(card, command="pass"))
    assert not ok and code == 409 and "idempotency" in result["reason"]


def test_worker_only_enables_reviewed_actions_for_explicit_nonmanual_source():
    class Driver:
        def observe_pass(self, **_kwargs): pass
        def observe_open_targeted_like(self, *_args, **_kwargs): pass
        def observe_send_targeted_like(self, *_args, **_kwargs): pass

    manual = Worker("hinge", Driver(), None, None, None, "r", None, threading.Event(),
                    observe_source="manual")
    reviewed = Worker("hinge", Driver(), None, None, None, "r", None, threading.Event(),
                      observe_source="automation")
    assert not manual.observe_action_supported
    assert reviewed.observe_action_supported


def test_reviewed_worker_uses_retained_anchor_capture_only_for_explicit_bridge_source():
    class Driver:
        def observe_pass(self, **_kwargs): pass
        def observe_open_targeted_like(self, *_args, **_kwargs): pass
        def observe_send_targeted_like(self, *_args, **_kwargs): pass
        def current_profile(self): pass
        def current_profile_reviewed(self): pass

    manual = Worker("hinge", Driver(), None, None, None, "r", None, threading.Event(),
                    observe_source="manual")
    unbridged = Worker("hinge", Driver(), None, None, None, "r", None, threading.Event(),
                       observe_source="external_ai_review")
    reviewed = Worker("hinge", Driver(), None, None, None, "r", None, threading.Event(),
                      observe_source="external_ai_review",
                      observe_action_bridge=ObserveActionBridge())
    assert manual._observe_capture_method() == "current_profile"
    assert unbridged._observe_capture_method() == "current_profile"
    assert reviewed._observe_capture_method() == "current_profile_reviewed"


def test_worker_open_marks_sheet_open_only_after_driver_verification_fact():
    calls = []

    class Driver:
        def observe_pass(self, *, should_stop):
            calls.append(("pass", should_stop()))

        def observe_open_targeted_like(self, item, *, should_stop):
            calls.append(("open", item, should_stop()))

        def observe_send_targeted_like(self, text, item, *, should_stop):
            calls.append(("send", text, item, should_stop()))

    bridge, worker, card = ObserveActionBridge(), None, None
    worker = Worker("hinge", Driver(), None, None, None, "run-1", None, threading.Event(),
                    observe_source="automation", observe_action_bridge=bridge)
    bridge.register(worker)
    card = bridge.begin_card(worker)
    bridge.update_suggestion(worker, card["profile_token"], SimpleNamespace(index=3, text="hello"))
    bridge.mark_pre_tap_published(worker, card["profile_token"])
    assert bridge.submit(_request(card))[0]
    action = bridge.claim(worker, card["profile_token"])
    outcome = worker._perform_observe_action(action, SimpleNamespace(_pick=SimpleNamespace(index=3, text="hello")))
    assert outcome == "continue"
    assert calls == [("open", 3, False)]
    assert bridge.snapshot()["checkpoints"][0]["phase"] == "sheet_open"


def test_worker_recovers_from_pre_tap_target_refusal_without_a_decision():
    """A refused reviewed open retires this card; it is neither a pass nor a like."""
    calls = []

    class Driver:
        supports_observe_like_intent = False

        def observe_pass(self, *, should_stop):
            calls.append("pass")

        def observe_open_targeted_like(self, item, *, should_stop):
            calls.append(("open", item))
            raise ItemTargetingError("target could not be reached", stage="navigate",
                                     intended=item, index_space="model_items")

        def observe_send_targeted_like(self, text, item, *, should_stop):
            calls.append(("send", text, item))

        def wait_for_decision(self, **_kwargs):
            raise AssertionError("the queued bridge action should be claimed before passive wait")

    bridge = ObserveActionBridge()
    worker = Worker("hinge", Driver(), None, None, None, "run-1", None, threading.Event(),
                    observe_source="external_ai_review", observe_action_bridge=bridge)
    bridge.register(worker)
    card = bridge.begin_card(worker)
    bridge.update_suggestion(worker, card["profile_token"], SimpleNamespace(index=3, text="hello"))
    bridge.mark_pre_tap_published(worker, card["profile_token"])
    assert bridge.submit(_request(card))[0]

    outcome = worker._wait_for_observed_decision(
        SimpleNamespace(_pick=SimpleNamespace(index=3, text="hello")), card["profile_token"])

    assert outcome is _OBSERVE_PRE_TAP_TARGETING_REFUSED
    assert calls == [("open", 3)]
    snapshot = bridge.snapshot(run_id="run-1", app="hinge")
    assert snapshot["checkpoints"][0]["phase"] == "terminal"
    result = snapshot["results"][-1]
    assert result["command"] == "targeted_like_open"
    assert result["status"] == "failed"
    assert "ItemTargetingError" in result["reason"]
    # The owning loop's finally block ends the card before its normal recapture.
    bridge.end_card(worker, card["profile_token"])
    assert bridge.snapshot(run_id="run-1", app="hinge")["checkpoints"] == []


def test_worker_keeps_post_tap_targeting_refusal_fail_closed():
    """A sheet verification refusal may follow a heart tap, so it must still halt."""
    class Driver:
        def observe_pass(self, *, should_stop): pass

        def observe_open_targeted_like(self, item, *, should_stop):
            raise ItemTargetingError("opened sheet did not match", stage="verify", intended=item,
                                     actual=2, index_space="model_items")

        def observe_send_targeted_like(self, text, item, *, should_stop): pass

    bridge = ObserveActionBridge()
    worker = Worker("hinge", Driver(), None, None, None, "run-1", None, threading.Event(),
                    observe_source="external_ai_review", observe_action_bridge=bridge)
    bridge.register(worker)
    card = bridge.begin_card(worker)
    bridge.update_suggestion(worker, card["profile_token"], SimpleNamespace(index=3, text="hello"))
    bridge.mark_pre_tap_published(worker, card["profile_token"])
    assert bridge.submit(_request(card))[0]
    action = bridge.claim(worker, card["profile_token"])

    with pytest.raises(ItemTargetingError, match="opened sheet"):
        worker._perform_observe_action(action, SimpleNamespace(_pick=SimpleNamespace(index=3, text="hello")))

    checkpoint = bridge.snapshot(run_id="run-1", app="hinge")["checkpoints"][0]
    assert checkpoint["phase"] == "terminal"
    assert bridge.snapshot(run_id="run-1", app="hinge")["results"][-1]["status"] == "failed"


def test_card_end_or_worker_exit_aborts_pending_and_snapshot_exposes_result():
    bridge, worker, card = _bound_card()
    assert bridge.submit(_request(card))[0]
    bridge.end_card(worker, card["profile_token"])
    snap = bridge.snapshot(run_id="run-1", app="hinge")
    assert snap["checkpoints"] == []
    assert snap["results"][-1]["status"] == "aborted"


def test_pass_has_no_item_and_unsupported_worker_cannot_be_approved():
    bridge, worker, card = _bound_card()
    ok, result, code = bridge.submit(_request(card, command="pass", item=3))
    assert not ok and code == 400
    worker.observe_action_supported = False
    ok, result, code = bridge.submit(_request(card, command="pass"))
    assert not ok and code == 409 and "cannot safely" in result["reason"]
