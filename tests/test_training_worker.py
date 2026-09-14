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
from operation_love.opener.opener import INDEX_SPACE_MODEL_ITEMS, INDEX_SPACE_PROFILE_PHOTOS
from operation_love.opener.service import OpenerPick
from operation_love.perception.capture import Profile
from operation_love.ranker.profile_key import profile_key_from_identity
from operation_love.status import RunStatus
from operation_love.training_actions import TrainingActionBridge
from operation_love.worker import Worker
from operation_love.limits import RateLimiter

from types import SimpleNamespace

# See tests/test_worker.py's own _FAKE_IDENTITY for why a SimpleNamespace is a valid stand-in
# for drivers.item_identity.ProfileIdentity here (profile_key_from_identity duck-types it).
_FAKE_IDENTITY = SimpleNamespace(known=True, fingerprint=(10, 20, 30), grid=(64, 16))
_FAKE_PROFILE_KEY = profile_key_from_identity(_FAKE_IDENTITY)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))


_FRAME = (b"\x89PNG\r\n\x1a\n"
          + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
          + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00"))
          + _png_chunk(b"IEND", b""))
_TIMEOUT_S = 5.0


@pytest.fixture(autouse=True)
def _stub_native_training_notification(monkeypatch):
    """Worker tests must never display real host notifications."""
    monkeypatch.setattr(worker_module, "notify_training_decision_ready", lambda: True)


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
        self.discards = []
        self.discard_result = True
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

    def discard_opener(self, pick, **lineage):
        self.discards.append((pick, lineage))
        self.events.append(("discard", lineage.get("decision")))
        return self.discard_result


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
        self.failure_snapshots = []

    def set_auto_session_policy(self, policy):
        self.policies.append(policy)

    def set_opener_enabled(self, enabled):
        self.opener_enabled.append(enabled)

    def set_training_decision(self, callback):
        self._decision = callback

    def render_status(self, _snapshot):
        pass

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
        return Profile(photos=[_FRAME], name="Ari", items=(b"item one", b"item two"))

    def current_profile_identity(self):
        return _FAKE_IDENTITY

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

    def snapshot_failure(self, exc):
        self.failure_snapshots.append(exc)

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
                archive_result=True, flush_error=None, fail_retrain=False, limiter=None,
                status=None):
    events = []
    driver = _TrainingDriver(events, after_choice=driver_after_choice)
    decider = _Decider(events, fail_retrain=fail_retrain)
    opener = _Opener(events)
    store = _Store(events, fail_profile=fail_profile, archive_result=archive_result,
                   flush_error=flush_error)
    bridge = bridge or _RecordingBridge(events)
    worker = Worker("hinge", driver, decider, opener, store, "training-run", _Pacing(),
                    threading.Event(), mode="training", training_action_bridge=bridge,
                    limiter=limiter, status=status)
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
        # profile_key (2026-09-06): read off the driver's current_profile_identity() BEFORE
        # like() is ever called (see worker.py's _current_profile_key), so a committed Like
        # carries the real attribution key, not "" or a stale value.
        assert opener.commits[0][1]["profile_key"] == _FAKE_PROFILE_KEY
    # The other half of the same lifecycle (discard_opener's own docstring): a reviewed
    # Dislike must not just drop this staged, model-generated draft the way it always used
    # to -- the durable table needs a `decision="dislike"` row instead of nothing, carrying
    # the SAME manual lineage (profile_id/decision_created_at) the Like branch above would
    # have carried had the Hub instead chosen Like.
    assert len(opener.discards) == (0 if command == "like" else 1)
    if command == "dislike":
        discard_lineage = opener.discards[0][1]
        assert discard_lineage["decision"] == "dislike"
        # SAME profile, SAME key a committed Like would have carried (see the `if command ==
        # "like"` branch above) -- the never_sent population is exactly as attributable as
        # the sent one.
        assert discard_lineage["profile_key"] == _FAKE_PROFILE_KEY
        assert discard_lineage["decision_source"] == "manual"
        assert discard_lineage["profile_id"] == store.profiles[0][0]
        assert isinstance(discard_lineage["decision_created_at"], float)
    assert events.index(("label", command == "like", "manual")) < events.index(
        ("hub_complete", "completed"))
    assert events.index(("flush",)) < events.index(("hub_complete", "completed"))
    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert (result["command"], result["status"]) == (command, "completed")


def test_training_card_carries_the_media_ordinal_its_driver_could_count():
    """The worker seam is the one place holding BOTH the live driver and the model's pick, so it
    is where the reviewer's photo/video position is counted and attached to the card."""
    worker, driver, _decider, _opener, _store, bridge, _events = _new_worker()
    asked = []

    def count(model_item_index):
        asked.append(model_item_index)
        return 4

    driver.model_item_media_ordinal = count
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))

    assert card["item_media_ordinal"] == 4
    # The MODEL ITEM number the opener was written about, never a heart ordinal or frame index.
    assert asked == [2]

    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()


def _refuses(_model_item_index):
    raise RuntimeError("a broken optional review hint must not reach the checkpoint")


@pytest.mark.parametrize("hook", [
    None,                          # a driver that never had the affordance at all
    lambda _index: None,           # fail-closed: the capture could not prove the count
    lambda _index: 0,              # never a 1-based position
    lambda _index: "3",            # the protocol carries an integer or nothing
    _refuses,                      # a broken optional implementation
])
def test_a_refused_media_ordinal_never_delays_alters_or_halts_the_checkpoint(hook):
    """NEVER BLOCKING. Whatever the hint does, the checkpoint that gets published is the one that
    would have been published without it, the operator still decides, and the run still lands and
    records -- a missing review hint is not a stop condition."""
    worker, driver, _decider, _opener, store, bridge, _events = _new_worker()
    if hook is not None:
        driver.model_item_media_ordinal = hook
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))

    assert "item_media_ordinal" not in card
    assert card["phase"] == "waiting_training_decision" and card["action"] == "ready"
    assert card["opener"] == "A precise typed opener" and card["item"] == 2

    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()
    assert not worker.stop_event.is_set()
    assert store.decisions[0][0:3] == ("like", 1.0, "manual")


def test_the_media_ordinal_hook_is_never_asked_about_a_legacy_profile_photo_pick():
    """The hook counts NUMBERED ITEM crops. A legacy pick's integer indexes raw capture frames
    instead, so handing it over would count a different card entirely -- the same reasoning
    `_item_type_preflight_mismatch` already applies to doc 5.8's preflight."""
    asked = []
    driver = type("Driver", (), {
        "model_item_media_ordinal": lambda _self, index: asked.append(index) or 1})()
    legacy = OpenerPick(text="opener", index=2, referenced="a photo",
                        item_description="a photo",
                        index_space=INDEX_SPACE_PROFILE_PHOTOS)
    model_pick = OpenerPick(text="opener", index=2, referenced="a photo",
                            item_description="a photo",
                            index_space=INDEX_SPACE_MODEL_ITEMS)

    assert worker_module._model_item_media_ordinal(driver, legacy) is None
    assert asked == []
    assert worker_module._model_item_media_ordinal(driver, model_pick) == 1
    assert asked == [2]


def test_training_status_says_the_phone_action_landed_while_archive_and_flush_run():
    """A slow durable write must not leave the Hub implying the phone tap is still pending."""
    observed = []
    status = RunStatus("training-run", ["hinge"], min_labels=1, mode="training")

    class _ProgressStore(_Store):
        def record_profile(self, run_id, app, profile_id, liked, source="auto", *, progress=None,
                           **metadata):
            assert callable(progress)
            progress("profile_upload", 1, 2)
            observed.append(("archive", status.app_view("hinge")["app"].copy()))
            progress("profile_uploaded", 2, 2)
            observed.append(("archive_complete", status.app_view("hinge")["app"].copy()))
            return super().record_profile(
                run_id, app, profile_id, liked, source=source, **metadata)

        def flush(self):
            observed.append(("flush", status.app_view("hinge")["app"].copy()))
            super().flush()

    worker, driver, decider, opener, _store, bridge, events = _new_worker(status=status)
    store = _ProgressStore(events)
    worker.store = store
    # The retrain that follows a completed action is the first thing the worker does after the
    # Hub result is durable, and nothing between it and the next capture publishes status -- so
    # sampling here reads exactly what the operator sees for the whole retrain/pace/break window.
    retrain = decider.retrain
    decider.retrain = lambda store_arg: (
        observed.append(("after_completion", status.app_view("hinge")["app"].copy()))
        or retrain(store_arg))
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)

    assert not worker.is_alive()
    archive = dict(observed)["archive"]
    archive_complete = dict(observed)["archive_complete"]
    flushing = dict(observed)["flush"]
    assert archive["state"] == archive_complete["state"] == flushing["state"] == "acting"
    assert archive["detail"] == "Like landed in Hinge; archiving profile screenshots (1/2)"
    assert archive_complete["detail"] == (
        "Like landed in Hinge; profile archive complete—recording the training label")
    assert flushing["detail"] == (
        "Like landed in Hinge; archive complete—flushing the label and evidence to storage")
    # …and the in-flight claim must not outlive the write it describes.  The Hub renders an
    # ``acting`` state as a wait box either way, so clearing only the detail would still assert
    # an unfinished device action while the worker is merely pacing.
    settled = dict(observed)["after_completion"]
    assert settled["state"] == "scoring"
    assert settled["detail"] == "Like recorded and saved; pacing before the next profile"


class _LegacyProtocolStore(_Store):
    """A store carrying LocalStore's exact ``record_profile`` signature.

    No ``progress`` parameter and no ``**kwargs``: the sqlite backend is written exactly this
    way, so handing it the optional progress callback is a TypeError raised AFTER the Hinge
    action has physically landed.
    """

    def record_profile(self, run_id, app, profile_id, liked, source="manual", photos=None,
                       photo_count=0, capture_truncated=False):
        self.events.append(("profile", liked, source))
        self.profiles.append((profile_id, liked, source, {
            "photo_count": photo_count, "capture_truncated": capture_truncated}))
        return self.archive_result


def test_training_archives_through_a_store_that_cannot_accept_a_progress_callback():
    """The sqlite-shaped store must never be offered the BigQuery-only progress keyword."""
    worker, driver, decider, opener, _store, bridge, events = _new_worker()
    store = _LegacyProtocolStore(events)
    worker.store = store
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)

    assert not worker.is_alive()
    assert store.profiles[0][1:3] == (True, "manual")
    assert store.labels and store.decisions
    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert result["status"] == "completed"


def test_progress_callback_is_never_leaked_through_a_legacy_metadata_seam():
    """A store that declares only ``**metadata`` accepts anything, which is why the guard has
    to be signature-based: an undeclared UI callback landing in a durable metadata bag is a
    silent contract violation rather than a visible error."""
    worker, driver, decider, opener, store, bridge, events = _new_worker()
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "dislike")
    worker.join(_TIMEOUT_S)

    assert not worker.is_alive()
    assert store.profiles and "progress" not in store.profiles[0][3]
    assert store.labels and "progress" not in store.labels[0][3]


def test_explicitly_accepts_keyword_admits_only_a_declared_parameter():
    """The deliberate asymmetry with ``_accepts_keywords``: an observational keyword is opt-in
    by declaration, so a ``**kwargs`` seam and an un-inspectable callable both decline it."""
    def declares(run_id, app, *, progress=None):
        pass

    def kwargs_only(run_id, app, **metadata):
        pass

    assert worker_module._explicitly_accepts_keyword(declares, "progress") is True
    assert worker_module._explicitly_accepts_keyword(kwargs_only, "progress") is False
    # Un-inspectable C callable: signature() raises, and the answer must be "no", unlike
    # _accepts_keywords which prefers the modern protocol for the same shape.
    assert worker_module._explicitly_accepts_keyword(dict.update, "progress") is False
    assert worker_module._accepts_keywords(dict.update, "progress") is True


@pytest.mark.parametrize("outcome", ["like", "dislike"])
@pytest.mark.parametrize("stage, counts, expected", [
    ("archive", (None, None), "archiving the reviewed profile and training data"),
    ("profile_upload", (1, 2), "archiving profile screenshots (1/2)"),
    ("profile_upload", (None, None), "archiving profile screenshots"),
    ("profile_uploaded", (2, 2), "profile archive complete—recording the training label"),
    ("opener_evidence", (None, None), "archiving the typed opener and send evidence"),
    ("label", (None, None), "recording the decision and training label"),
    ("flush", (None, None), "archive complete—flushing the label and evidence to storage"),
    ("some_future_stage", (None, None), "archiving the reviewed profile and training data"),
])
def test_persistence_detail_always_states_that_the_phone_action_already_landed(
        outcome, stage, counts, expected):
    status = RunStatus("training-run", ["hinge"], min_labels=1, mode="training")
    worker, *_rest = _new_worker(status=status)

    worker._training_persistence_status(outcome, stage, *counts)

    app = status.app_view("hinge")["app"]
    assert app["state"] == "acting"
    assert app["detail"] == f"{outcome.title()} landed in Hinge; {expected}"


@pytest.mark.parametrize("counts", [
    (True, True), (3, 2), (1, 0), (1, -1), ("1", 2), (None, 2), (1, None), (1.0, 2),
])
def test_nonsensical_store_progress_counts_fall_back_to_countless_wording(counts):
    """No shipped store can produce these, and that is the point: a store-supplied pair is
    rendered to the operator in the window right after an irreversible action landed, so a
    count that cannot be true must never reach the Hub as "(5/0)"."""
    status = RunStatus("training-run", ["hinge"], min_labels=1, mode="training")
    worker, *_rest = _new_worker(status=status)

    worker._training_persistence_status("like", "profile_upload", *counts)

    assert status.app_view("hinge")["app"]["detail"] == (
        "Like landed in Hinge; archiving profile screenshots")


def test_training_notifies_once_after_checkpoint_becomes_actionable(monkeypatch):
    notifications = []
    monkeypatch.setattr(
        worker_module, "notify_training_decision_ready",
        lambda: notifications.append(_checkpoint(bridge)),
    )
    worker, driver, decider, opener, store, bridge, events = _new_worker()
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _wait_until(lambda: notifications)
    _submit(bridge, card, "dislike")
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()

    # Counted over the whole run, not at the first arrival: the driver is a deliberate one-card
    # seam, so a second alert anywhere later in this profile's lifecycle (a re-publish retry, or
    # an alert repeated on the executing transition) must fail here rather than land after the
    # assertion has already run.
    assert len(notifications) == 1
    assert notifications[0]["profile_token"] == card["profile_token"]
    assert notifications[0]["pending"] is True


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


def test_stop_before_training_decision_is_not_captured_as_an_unexpected_failure():
    """A Hub Stop while the typed draft awaits review is an expected cancellation.

    The concrete Hinge driver raises ActionCancelled after its review callback returns the
    stop sentinel.  Keep that distinction at the worker loop boundary: it must close cleanly
    without asking the debug logger to create an ``unexpected`` error screenshot.
    """
    worker, driver, decider, opener, store, bridge, events = _new_worker()

    def stop_at_review(opener_text, item_index=None, *, model_item_index=None, should_stop=None):
        driver.like_calls.append((opener_text, item_index, model_item_index))
        assert driver._decision is not None
        assert driver._decision(_FRAME, {"evidence_id": "verified-composer"}) == "stop"
        raise ActionCancelled("action cancelled because the run is stopping before training decision")

    driver.like = stop_at_review
    worker.start()
    _wait_until(lambda: _checkpoint(bridge))
    worker.stop_event.set()
    worker.join(_TIMEOUT_S)

    assert not worker.is_alive()
    assert driver.failure_snapshots == []
    assert store.decisions == store.profiles == store.labels == []
    assert opener.commits == []
    assert decider.decide_calls == 0
    assert driver.closed


def test_training_item_index_refusal_closes_the_live_phone_session():
    """A safe pre-opener refusal still owns and releases its opened transport.

    An item-index contradiction is reported by Hinge as ``Profile.items_unavailable``;
    Training must stop before asking Gemini or installing a review action. The phone is
    deliberately left on the captured profile for diagnosis, but that must not be confused
    with keeping its session alive: ``Worker._finish_session`` has to close the driver, which
    is what unregisters the shipped persistent UHID touchscreen.
    """
    refusal = (
        "the item index this capture produced contradicts itself, so its numbering cannot "
        "be trusted")

    class _RefusingIndexDriver(_TrainingDriver):
        def next_profile(self):
            if self._profile_returned:
                return None
            self._profile_returned = True
            return Profile(photos=[_FRAME], name="Ari", items_unavailable=refusal)

    events = []
    driver = _RefusingIndexDriver(events)
    decider = _Decider(events)
    opener = _Opener(events)
    store = _Store(events)
    status = RunStatus("training-run", ["hinge"], min_labels=1, mode="training")
    stop = threading.Event()
    bridge = _RecordingBridge(events)
    worker = Worker("hinge", driver, decider, opener, store, "training-run", _Pacing(), stop,
                    mode="training", training_action_bridge=bridge, status=status)

    worker.run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "stopped"
    assert app["stop_kind"] == "opener"
    assert refusal in app["stop_reason"]
    assert stop.is_set()
    assert driver.opened and driver.closed
    assert driver._decision is None
    assert opener.maybe_calls == []
    assert driver.like_calls == []
    assert bridge.snapshot(run_id="training-run", app="hinge")["checkpoints"] == []


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
