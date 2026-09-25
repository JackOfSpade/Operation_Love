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
    # The OTHER half of that same lifecycle, once per abandonment handler around like()
    # (worker.py's `except ActionCancelled` and its generic `except Exception`): the provider
    # call already finished and was already BILLED, and nothing was sent, so this draft must
    # reach the durable table as a never_sent row instead of vanishing. Asserted here because
    # these two handlers are otherwise the only discard sites in _training_loop with no test
    # of their own -- both were replaced with `pass` in review and the whole suite stayed green.
    assert len(opener.discards) == 1
    discard_lineage = opener.discards[0][1]
    assert discard_lineage["decision"] == "never_sent"
    assert discard_lineage["decision_source"] == "manual"
    assert discard_lineage["profile_key"] == _FAKE_PROFILE_KEY
    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert result["status"] == expected
    assert events[-1] == ("hub_complete", expected)
    assert decider.decide_calls == 0


class _PostSendTrainingDriver(_TrainingDriver):
    """The real Training shape, which `_TrainingDriver` above deliberately is not.

    `driver.like()` is not atomic. Once the reviewer chooses Like, HingeDriver taps Send Like and
    only THEN demands a stable, semantically different ready deck card -- raising
    HingeActionError when it cannot get one, with the opener already sitting in front of a real
    person. `_TrainingDriver` raises BEFORE that tap (and reports no send marker at all), which
    is why it pins the never_sent direction and cannot pin this one.

    `after_choice` keeps its parent meaning and selects WHICH of the two post-send handlers the
    worker takes: "cancel" for a Stop landing between the tap and its verification, "fail" for
    the verification itself refusing.
    """

    def __init__(self, events, *, after_choice="fail"):
        super().__init__(events, after_choice=after_choice)
        self._send_attempted = False

    def like(self, opener, item_index=None, *, model_item_index=None, should_stop=None):
        # Cleared as the attempt begins, exactly as HingeDriver._like_comment_sheet does -- see
        # base.DatingAppDriver.like_send_attempted for why it is never cleared when one ends.
        self._send_attempted = False
        self.like_calls.append((opener, item_index, model_item_index))
        self.stop_callbacks.append(should_stop)
        self.events.append(("driver_like_called", opener, model_item_index))
        assert self._decision is not None
        command = self._decision(_FRAME, {"evidence_id": "verified-composer"})
        self.events.append(("driver_choice", command))
        self._send_attempted = True           # the Send Like tap: the opener is out
        if self.after_choice == "cancel":
            raise ActionCancelled("a Stop arrived between the Send Like tap and its verification")
        raise RuntimeError(
            "training action did not reach a stable, semantically different ready deck card")

    def like_send_attempted(self):
        return self._send_attempted


def _run_training_like_through(driver):
    """Drive one reviewed Like all the way to `driver.like()` and return the fake opener service.

    Built by hand rather than through `_new_worker` because the point of these tests is the
    driver, which `_new_worker` owns.
    """
    events = driver.events
    opener = _Opener(events)
    store = _Store(events)
    bridge = _RecordingBridge(events)
    status = RunStatus("training-run", ["hinge"], min_labels=1, mode="training")
    worker = Worker("hinge", driver, _Decider(events), opener, store, "training-run", _Pacing(),
                    threading.Event(), mode="training", training_action_bridge=bridge,
                    status=status)
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()
    return opener, store, bridge


@pytest.mark.parametrize("after_choice, expected", [
    ("cancel", "aborted"),
    ("fail", "failed"),
])
def test_a_post_send_training_failure_records_the_draft_as_send_unverified(
        after_choice, expected):
    """The reviewer said Like, the phone sent it, and the verification AFTER the send failed.

    Both abandonment handlers around `driver.like()` used to answer that with
    `decision="never_sent"` -- a durable row, in the one table the corpus report and the outcome
    join trust, saying an opener that physically went out was only ever a draft. The comment at
    the generic handler asserted "anything raised ... means no reviewed Like landed for this
    profile either": true, and not the same statement as the row it wrote. No label is created
    either way (nothing verified a Like), which is exactly why the opener row is the only place
    this fact can be recorded at all.
    """
    driver = _PostSendTrainingDriver([], after_choice=after_choice)
    opener, store, bridge = _run_training_like_through(driver)

    assert driver.like_calls == [("A precise typed opener", None, 2)]
    # Unchanged from the never_sent tests above: an unverified send is not a reviewed Like.
    assert store.decisions == store.profiles == store.labels == []
    assert opener.commits == []
    assert len(opener.discards) == 1
    lineage = opener.discards[0][1]
    assert lineage["decision"] == "send_unverified"
    assert lineage["decision_source"] == "manual"
    # Attributable, exactly like every other discarded draft -- both populations carry the key.
    assert lineage["profile_key"] == _FAKE_PROFILE_KEY
    assert bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]["status"] == expected
    # MUTATION CHECK, once per handler, because the two parametrizations exercise different ones:
    # drop `decision=self._post_like_discard_decision()` from _training_loop's `except
    # ActionCancelled` -- re-run: only the [cancel-aborted] case fails; drop it from the generic
    # `except Exception` -- re-run: only [fail-failed] fails. Verified by hand, restored exactly.


@pytest.mark.parametrize("after_choice", ["cancel", "fail"])
def test_a_training_failure_before_the_send_still_records_never_sent(after_choice):
    """The other direction through the SAME two handlers, and the reason it is pinned here.

    `_TrainingDriver` raises before it ever taps Send, and -- being a duck-typed fake that does
    not subclass DatingAppDriver at all -- carries no send marker whatsoever, so this also pins
    the worker's defensive `getattr` read: an optional capability missing entirely must degrade
    to today's meaning rather than raise an AttributeError inside a live exception handler. A
    fix that filed every post-like failure as `send_unverified` would be just as wrong as the one
    it replaced, in the other direction.
    """
    driver = _TrainingDriver([], after_choice=after_choice)
    opener, store, bridge = _run_training_like_through(driver)

    assert driver.like_calls == [("A precise typed opener", None, 2)]
    assert len(opener.discards) == 1
    assert opener.discards[0][1]["decision"] == "never_sent"
    assert opener.discards[0][1]["decision_source"] == "manual"
    # MUTATION CHECK: collapse _post_like_discard_decision() to `return DECISION_SEND_UNVERIFIED`
    # -- re-run: both parametrizations here fail, as do every other never_sent assertion in this
    # file. Verified by hand, restored exactly.


def test_a_blank_generated_opener_is_still_recorded_as_a_billed_never_sent_draft():
    """A pick whose TEXT is unusable stops the run -- and is exactly the row the corpus report
    most needs to see.

    The provider call already finished and was already billed by the time the worker looks at
    the text, so silence here would put the WORST drafts back in the survivorship-bias hole
    discard_opener exists to close. No checkpoint is ever published on this path, so the run is
    driven synchronously.
    """
    status = RunStatus("training-run", ["hinge"], min_labels=1, mode="training")
    worker, driver, decider, opener, store, bridge, events = _new_worker(status=status)
    opener.pick = OpenerPick(text="   ", index=2, referenced="the hiking photo",
                             item_description="hiking photo",
                             index_space=INDEX_SPACE_MODEL_ITEMS)

    worker.run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "stopped" and app["stop_kind"] == "opener"
    assert driver.like_calls == [] and store.decisions == store.profiles == store.labels == []
    assert opener.commits == []
    assert len(opener.discards) == 1
    pick, lineage = opener.discards[0]
    assert pick is opener.pick
    assert lineage["decision"] == "never_sent" and lineage["decision_source"] == "manual"
    assert lineage["profile_key"] == _FAKE_PROFILE_KEY
    assert bridge.snapshot(run_id="training-run", app="hinge")["checkpoints"] == []


class _SpentLikeBudget(RateLimiter):
    """A run whose ``max_likes_per_run`` allowance is already gone when the next draft lands.

    Subclassed rather than hand-rolled so every other part of the limiter contract the worker
    touches (``allow``/``max_per_day``) stays the real one; only the like budget is exhausted.
    """
    def allow_like(self, liked_this_run):
        return False


def test_a_rate_limited_run_still_records_the_draft_it_abandoned():
    """The like allowance runs out AFTER this profile's draft was generated and billed.

    A rate-limited run must not be a hole in the opener corpus: the refusal happens before any
    checkpoint is published, so without the discard the draft reaches neither the durable
    `openers` table nor recent_openers.
    """
    status = RunStatus("training-run", ["hinge"], min_labels=1, mode="training")
    worker, driver, decider, opener, store, bridge, events = _new_worker(
        limiter=_SpentLikeBudget(max_likes_per_run=1), status=status)

    worker.run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "rate_limited"
    assert driver.like_calls == [] and store.decisions == store.profiles == store.labels == []
    assert opener.commits == []
    assert len(opener.discards) == 1
    lineage = opener.discards[0][1]
    assert lineage["decision"] == "never_sent" and lineage["decision_source"] == "manual"
    assert lineage["profile_key"] == _FAKE_PROFILE_KEY
    assert bridge.snapshot(run_id="training-run", app="hinge")["checkpoints"] == []


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


def test_training_stops_for_no_opener_consumer_unlike_auto():
    """Training hard-stops on ``items_unavailable_kind == "no_opener_consumer"`` -- the same
    generic ``items_unavailable`` stop the untyped-kind case above exercises -- even though
    AUTO's own items_unavailable check is guarded by ``accepts_opener and not disabled`` and
    never even reaches this kind while openers are off (see worker.py's AUTO loop, and the
    comment above this Training branch at worker.py naming both consumers of this kind).

    This is the third consumer of the ``no_opener_consumer`` string, alongside hinge.py (which
    stamps it) and bugreport.py's completion-verdict layer (which reads it to decline degrading
    the run's verdict for AUTO's configured-off bare-like). Before this test, worker.py's own
    reading of the kind for Training was implicit and unpinned: a future change that added a
    blanket ``no_opener_consumer`` skip to Training too -- copying AUTO's exemption without
    noticing Training has no bare-like fallback to fall back to -- would leave Training silently
    producing no opener and no stop, with nothing catching the regression.

    The decision (see worker.py's dated comment on this branch): with no opener consumer,
    Training -- whose whole purpose is preparing a typed opener for human review -- has nothing
    to prepare, so it stops, and the stop is reported through stop_kind="opener" rather than
    stop_kind="targeting_calibration" (that kind is reserved for a live calibration blocker).
    """
    refusal = "opener.enabled is false, so no numbered item list was requested"

    class _NoOpenerConsumerDriver(_TrainingDriver):
        def next_profile(self):
            if self._profile_returned:
                return None
            self._profile_returned = True
            return Profile(photos=[_FRAME], name="Ari", items_unavailable=refusal,
                           items_unavailable_kind="no_opener_consumer")

    events = []
    driver = _NoOpenerConsumerDriver(events)
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
    assert app["stop_kind"] == "opener"          # NOT "targeting_calibration"
    assert refusal in app["stop_reason"]
    assert stop.is_set()
    assert driver.opened and driver.closed
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
    # A replaced registration is REFUSED by the bridge while the prior worker still holds it,
    # so this arrives at the worker as a raise out of like() -- i.e. through the generic
    # abandonment handler, not the unverified-outcome branch below it. Either way nothing was
    # typed or sent, so the billed draft must reach the durable table as never_sent.
    assert len(opener.discards) == 1
    assert opener.discards[0][1]["decision"] == "never_sent"
    assert opener.discards[0][1]["decision_source"] == "manual"

    # With no Hub mailbox at all, training fails closed before opening a session or generating.
    worker, driver, decider, opener, store, bridge, events = _new_worker()
    worker.training_action_bridge = None
    worker.start()
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()
    assert not driver.opened and opener.maybe_calls == []
    assert store.decisions == store.profiles == store.labels == []


class _SilentBridge(_RecordingBridge):
    """A Hub mailbox whose decision window closes with no answer (the ordinary Stop)."""
    def wait_for_action(self, worker, profile_token, stop_event):
        return None


def test_an_unverified_driver_outcome_records_the_draft_nothing_was_sent_for():
    """The outcome half of `outcome not in {"like","dislike"} or action is None`.

    No answer ever arrives, so the checkpoint is cancelled and the driver hands back the stop
    sentinel instead of a verified Like/Dislike. The driver contract leaves the phone untouched
    on that path -- nothing was typed or sent -- so this billed draft is exactly the kind of row
    the corpus report was blind to. Distinct from the abandonment handlers above it: this one
    reaches the branch without any exception at all, which is why those tests cannot pin it.
    """
    status = RunStatus("training-run", ["hinge"], min_labels=1, mode="training")
    events = []
    driver = _TrainingDriver(events)
    opener = _Opener(events)
    store = _Store(events)
    bridge = _SilentBridge(events)
    worker = Worker("hinge", driver, _Decider(events), opener, store, "training-run", _Pacing(),
                    threading.Event(), mode="training", training_action_bridge=bridge,
                    status=status)

    worker.run()

    app = status.app_view("hinge")["app"]
    # A clean Stop, never the RuntimeError: the run breaks out with the stop flag already set.
    assert app["state"] == "stopped" and not app.get("error")
    assert driver.like_calls == [("A precise typed opener", None, 2)]
    assert store.decisions == store.profiles == store.labels == []
    assert opener.commits == []
    assert len(opener.discards) == 1
    lineage = opener.discards[0][1]
    assert lineage["decision"] == "never_sent" and lineage["decision_source"] == "manual"
    assert lineage["profile_key"] == _FAKE_PROFILE_KEY


def test_a_landed_like_with_no_hub_claim_raises_and_is_never_written_down_as_never_sent():
    """The `action is None` half of the unverified-outcome branch means the OPPOSITE of the
    outcome half, so it must not share its discard.

    A driver that returns "like" has typed the opener and physically SENT it; what went missing
    is the Hub's claim for it (a stale or replaced registration). The branch is a DISJUNCTION,
    so it used to run `_discard_staged_opener` here too and write a durable
    `decision="never_sent"` opener row for an opener that actually landed -- wrong data in the
    one table the corpus report trusts, published moments before the RuntimeError. An
    unattributed sent opener is recoverable; a row asserting it was never sent is not.
    """
    status = RunStatus("training-run", ["hinge"], min_labels=1, mode="training")

    class _UnclaimedSendDriver(_TrainingDriver):
        def like(self, opener_text, item_index=None, *, model_item_index=None, should_stop=None):
            # The phone did the Like. The worker simply never saw a claimed Hub action for it.
            self.like_calls.append((opener_text, item_index, model_item_index))
            return "like"

    events = []
    driver = _UnclaimedSendDriver(events)
    decider = _Decider(events)
    opener = _Opener(events)
    store = _Store(events)
    bridge = _RecordingBridge(events)
    worker = Worker("hinge", driver, decider, opener, store, "training-run", _Pacing(),
                    threading.Event(), mode="training", training_action_bridge=bridge,
                    status=status)

    worker.run()

    app = status.app_view("hinge")["app"]
    # run() HALTS on the RuntimeError rather than swallowing it (see Worker.run's own comment).
    assert app["state"] == "error"
    assert "no verified Like/Dislike outcome" in app["error"]
    assert driver.like_calls == [("A precise typed opener", None, 2)]
    # The whole point: no decision row of any kind was written for a draft that went out.
    assert opener.discards == [] and opener.commits == []
    assert store.decisions == store.profiles == store.labels == []
    assert ("discard", "never_sent") not in events


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


def test_training_none_capture_publishes_a_failed_cold_relaunch_under_its_own_stop_kind(
        monkeypatch):
    """Training's own capture-returned-None probe classifies the latch exactly like AUTO's.

    Same incident as tests/test_worker.py's AUTO counterpart: `blocked_reason()` is the only
    channel a capture that returned None has for carrying a sentence, so a failed cold-relaunch
    recovery came out of it too and the Hub headlined "stopped -- deck blocked" for a deck that
    was never blocked. Training must not disagree with AUTO about that.

    Runs against status.py's REAL `_STOP_KINDS`. It was briefly monkeypatched here while the
    whitelist entry was still missing, which made this test pass over a path that raised
    ValueError out of set_app in production -- the widening is the fix, not a test fixture.
    """
    reason = ("Hinge was relaunched cold and the deck could not be proven re-entered; "
              "no action or label was recorded.")

    class _ColdRelaunchDriver(_TrainingDriver):
        def __init__(self, events):
            super().__init__(events)
            self.abandoned = False

        def blocked_reason(self):
            return reason if self.abandoned else None

        def blocked_stop_kind(self):
            return "cold_relaunch_recovery" if self.abandoned else None

        def next_profile(self):
            self.abandoned = True
            return None

    status = RunStatus("training-run", ["hinge"], min_labels=1, mode="training")
    events = []
    driver = _ColdRelaunchDriver(events)
    opener = _Opener(events)
    store = _Store(events)
    bridge = _RecordingBridge(events)
    worker = Worker("hinge", driver, _Decider(events), opener, store, "training-run", _Pacing(),
                    threading.Event(), mode="training", training_action_bridge=bridge,
                    status=status)

    worker.run()

    app = status.app_view("hinge")["app"]
    assert app["state"] == "blocked" and app["stop_reason"] == reason
    assert app["stop_kind"] == "cold_relaunch_recovery"
    # Nothing was generated, typed or acted on: this is the capture probe, not an action path.
    assert opener.maybe_calls == [] and opener.discards == [] and driver.like_calls == []
    assert store.decisions == store.profiles == store.labels == []
    assert driver.opened and driver.closed
