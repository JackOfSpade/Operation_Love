"""End-to-end worker ownership tests for Hub-reviewed training.

These use the real :class:`Worker` and :class:`TrainingActionBridge`; only the
phone, opener provider, ranker, and durable store are faked.  In particular a
Hub command is not considered complete until the worker has recorded it.
"""
from __future__ import annotations

import struct
import threading
import time
import zlib

import pytest

import operation_love.worker as worker_module
from operation_love.drivers.base import ActionCancelled
from operation_love.opener.opener import INDEX_SPACE_MODEL_ITEMS
from operation_love.opener.service import OpenerPick
from operation_love.perception.capture import Profile
from operation_love.training_actions import TrainingActionBridge
from operation_love.worker import Worker
from operation_love.limits import RateLimiter


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))


_FRAME = (b"\x89PNG\r\n\x1a\n"
          + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
          + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00"))
          + _png_chunk(b"IEND", b""))
_TIMEOUT_S = 5.0


class _Pacing:
    swipe_delay_s = 0.0


class _Decider:
    def __init__(self, events, *, fail_retrain=False):
        self.events = events
        self.fail_retrain = fail_retrain
        self.decide_calls = 0

    def decide(self, profile):
        self.decide_calls += 1
        raise AssertionError("training must never ask the ranker to decide")

    def embed(self, profile):
        self.events.append(("embed", profile.name))
        return [0.25, 0.75]

    def retrain(self, store):
        self.events.append(("retrain",))
        if self.fail_retrain:
            raise RuntimeError("ranker retrain unavailable")
        return True


class _Store:
    def __init__(self, events, *, fail_profile=False, archive_result=True,
                 flush_error=None):
        self.events = events
        self.fail_profile = fail_profile
        self.archive_result = archive_result
        self.flush_error = flush_error
        self.decisions = []
        self.profiles = []
        self.labels = []
        self.count_calls = []

    def count_today(self, app, *, source="auto"):
        self.count_calls.append((app, source))
        return 0

    def flush(self):
        self.events.append(("flush",))
        if self.flush_error is not None:
            raise self.flush_error

    def record_decision(self, run_id, app, decision, score, source="auto", **lineage):
        self.decisions.append((decision, score, source, lineage))
        self.events.append(("decision", decision, source))

    def record_profile(self, run_id, app, profile_id, liked, source="auto", **metadata):
        self.events.append(("profile", liked, source))
        if self.fail_profile:
            raise RuntimeError("profile archive unavailable")
        self.profiles.append((profile_id, liked, source, metadata))
        return self.archive_result

    def add_label(self, run_id, app, liked, embedding, source="auto", **metadata):
        self.labels.append((liked, embedding, source, metadata))
        self.events.append(("label", liked, source))


class _Opener:
    disabled = False
    stop_requested = False

    def __init__(self, events):
        self.events = events
        self.maybe_calls = []
        self.commits = []
        self.commit_result = True
        self.pick = OpenerPick(text="A precise typed opener", index=2,
                               referenced="the hiking photo", item_description="hiking photo",
                               index_space=INDEX_SPACE_MODEL_ITEMS)

    def maybe_opener(self, run_id, app, profile, *, items=None, should_stop=None, stage=False):
        self.maybe_calls.append((run_id, app, profile, items, should_stop, stage))
        self.events.append(("generated", self.pick.text, self.pick.index))
        return self.pick

    def commit_opener(self, pick, **lineage):
        self.commits.append((pick, lineage))
        self.events.append(("commit", lineage.get("decision")))
        return self.commit_result


class _TrainingDriver:
    """A one-card Hinge seam which invokes the driver's installed decision hook."""
    accepts_opener = True
    supports_training_decision = True
    supports_interruptible_like_navigation = True

    def __init__(self, events, *, after_choice="return"):
        self.events = events
        self.after_choice = after_choice
        self._decision = None
        self._profile_returned = False
        self.opened = False
        self.closed = False
        self.policies = []
        self.opener_enabled = []
        self.like_calls = []
        self.stop_callbacks = []

    def set_auto_session_policy(self, policy):
        self.policies.append(policy)

    def set_opener_enabled(self, enabled):
        self.opener_enabled.append(enabled)

    def set_training_decision(self, callback):
        self._decision = callback

    def open_session(self):
        self.opened = True

    def blocked_reason(self):
        return None

    def out_of_profiles(self):
        return self._profile_returned

    def next_profile(self):
        if self._profile_returned:
            return None
        self._profile_returned = True
        return Profile(photos=[b"photo"], name="Ari", items=(b"item one", b"item two"))

    def like(self, opener, item_index=None, *, model_item_index=None, should_stop=None):
        self.like_calls.append((opener, item_index, model_item_index))
        self.stop_callbacks.append(should_stop)
        self.events.append(("driver_like_called", opener, model_item_index))
        assert self._decision is not None
        command = self._decision(_FRAME, {"evidence_id": "verified-composer"})
        self.events.append(("driver_choice", command))
        if self.after_choice == "cancel":
            raise ActionCancelled("stop at device boundary")
        if self.after_choice == "fail":
            raise RuntimeError("device action failed")
        return command

    def landed_auto_opener_evidence(self):
        return {"evidence_id": "verified-landed-composer"}

    def close(self):
        self.closed = True


class _RecordingBridge(TrainingActionBridge):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def complete(self, action, status, reason=None):
        self.events.append(("hub_complete", status))
        return super().complete(action, status, reason)


class _ReplacingBridge(_RecordingBridge):
    """Makes the worker stale while it is waiting, before a device choice can land."""
    def wait_for_action(self, worker, profile_token, stop_event):
        replacement = type("Replacement", (), {
            "run_id": worker.run_id, "app": worker.app,
            "training_action_supported": True, "stop_event": threading.Event(),
        })()
        self.register(replacement)
        return None


def _new_worker(*, bridge=None, driver_after_choice="return", fail_profile=False,
                archive_result=True, flush_error=None, fail_retrain=False, limiter=None):
    events = []
    driver = _TrainingDriver(events, after_choice=driver_after_choice)
    decider = _Decider(events, fail_retrain=fail_retrain)
    opener = _Opener(events)
    store = _Store(events, fail_profile=fail_profile, archive_result=archive_result,
                   flush_error=flush_error)
    bridge = bridge or _RecordingBridge(events)
    worker = Worker("hinge", driver, decider, opener, store, "training-run", _Pacing(),
                    threading.Event(), mode="training", training_action_bridge=bridge,
                    limiter=limiter)
    return worker, driver, decider, opener, store, bridge, events


def _wait_until(predicate):
    deadline = time.monotonic() + _TIMEOUT_S
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    pytest.fail("timed out waiting for training worker")


def _checkpoint(bridge):
    snapshot = bridge.snapshot(run_id="training-run", app="hinge")
    return snapshot["checkpoints"][0] if snapshot["checkpoints"] else None


def _submit(bridge, card, command):
    ok, result, code = bridge.submit({
        "command": command, "run_id": card["run_id"], "app": card["app"],
        "profile_token": card["profile_token"], "approval_token": card["approval_token"],
        "idempotency_token": f"{command}-request",
    })
    assert (ok, code, result["status"]) == (True, 202, "queued")


@pytest.mark.parametrize("command", ["like", "dislike"])
def test_training_persists_verified_human_choice_only_after_hub_choice(command, monkeypatch):
    class _ForbiddenAutoPolicy:
        def __init__(self, *args, **kwargs):
            raise AssertionError("training must not construct AutoSessionPolicy")

    monkeypatch.setattr(worker_module, "AutoSessionPolicy", _ForbiddenAutoPolicy)
    worker, driver, decider, opener, store, bridge, events = _new_worker()
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))

    # The model-generated, model-item-targeted opener has reached the device before the Hub
    # is asked to decide; no ranker decision or auto policy is involved.
    assert driver.like_calls == [("A precise typed opener", None, 2)]
    assert card["opener"] == "A precise typed opener"
    assert decider.decide_calls == 0
    assert driver.policies == [None]
    assert callable(driver.stop_callbacks[0]) and not driver.stop_callbacks[0]()

    _submit(bridge, card, command)
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()

    assert store.decisions[0][0:3] == (command, 1.0 if command == "like" else 0.0, "manual")
    assert store.profiles[0][1:3] == (command == "like", "manual")
    assert store.labels[0][0] is (command == "like")
    assert store.labels[0][2] == "manual"
    assert len(opener.commits) == (1 if command == "like" else 0)
    assert opener.commits == [] or opener.commits[0][1]["decision"] == "like"
    if command == "like":
        assert opener.commits[0][1]["pre_send_evidence"] == {
            "evidence_id": "verified-landed-composer"}
    assert events.index(("label", command == "like", "manual")) < events.index(
        ("hub_complete", "completed"))
    assert events.index(("flush",)) < events.index(("hub_complete", "completed"))
    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert (result["command"], result["status"]) == (command, "completed")


@pytest.mark.parametrize("after_choice, expected", [
    ("cancel", "aborted"),
    ("fail", "failed"),
])
def test_cancelled_or_failed_driver_action_never_creates_training_label(after_choice, expected):
    worker, driver, decider, opener, store, bridge, events = _new_worker(
        driver_after_choice=after_choice)
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)

    assert not worker.is_alive()
    assert store.decisions == store.profiles == store.labels == []
    assert opener.commits == []
    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert result["status"] == expected
    assert events[-1] == ("hub_complete", expected)
    assert decider.decide_calls == 0


def test_stale_or_absent_training_action_never_creates_label():
    events = []
    stale_bridge = _ReplacingBridge(events)
    worker, driver, decider, opener, store, bridge, events = _new_worker(bridge=stale_bridge)
    worker.start()
    _wait_until(lambda: driver.like_calls)
    worker.join(_TIMEOUT_S)

    assert not worker.is_alive()
    assert store.decisions == store.profiles == store.labels == []
    assert opener.commits == []
    assert decider.decide_calls == 0

    # With no Hub mailbox at all, training fails closed before opening a session or generating.
    worker, driver, decider, opener, store, bridge, events = _new_worker()
    worker.training_action_bridge = None
    worker.start()
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()
    assert not driver.opened and opener.maybe_calls == []
    assert store.decisions == store.profiles == store.labels == []


def test_persistence_failure_marks_hub_failed_and_never_completes_it():
    worker, driver, decider, opener, store, bridge, events = _new_worker(fail_profile=True)
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)

    assert not worker.is_alive()
    assert store.labels == [] and opener.commits == []
    assert store.decisions == []
    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert result["status"] == "failed"
    assert ("hub_complete", "completed") not in events
    assert events[-1] == ("hub_complete", "failed")


@pytest.mark.parametrize("kind", ["archive_false", "no_embedding", "flush", "opener"])
def test_incomplete_training_persistence_never_completes_the_hub_action(kind):
    worker, driver, decider, opener, store, bridge, events = _new_worker(
        archive_result=(kind != "archive_false"),
        flush_error=(RuntimeError("durable flush failed") if kind == "flush" else None))
    if kind == "no_embedding":
        decider.embed = lambda profile: None
    if kind == "opener":
        opener.commit_result = False
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)

    assert not worker.is_alive()
    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert result["status"] == "failed"
    assert ("hub_complete", "completed") not in events
    if kind in {"archive_false", "no_embedding", "opener"}:
        assert store.labels == []
    # Input validation/archive/commit failures are discovered before the manual decision row,
    # so they cannot consume a later Training run's daily decision allowance.
    if kind in {"archive_false", "no_embedding", "opener"}:
        assert store.decisions == []


def test_retrain_failure_after_durable_label_does_not_relabel_hub_action_failed():
    worker, driver, decider, opener, store, bridge, events = _new_worker(fail_retrain=True)
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "dislike")
    worker.join(_TIMEOUT_S)

    assert not worker.is_alive()
    assert store.labels and ("flush",) in events
    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert result["status"] == "completed"
    assert events.index(("hub_complete", "completed")) < events.index(("retrain",))


def test_training_daily_limit_reads_manual_history_not_auto_history():
    worker, driver, decider, opener, store, bridge, events = _new_worker(
        limiter=RateLimiter(max_per_day=2))
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    assert store.count_calls == [("hinge", "manual")]
    _submit(bridge, card, "dislike")
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()
