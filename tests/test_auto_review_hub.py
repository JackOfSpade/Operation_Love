"""AUTO pass review, hub side: validation, store plumbing, HTTP surface and the page's logic.

Why each group exists:

* validators -- an answer becomes a training label and a pass id reaches a store query / object
  path, so malformed ids and non-boolean answers must be refused at the boundary;
* store plumbing -- the card works with no run live (own store, closed after use), prefers the
  live run's store, never retries a WRITE, and flushes before it says "saved";
* HTTP -- the real handler over a real socket, including CSRF and the image headers;
* page -- the card's pure helpers and its answer flow are executed for real under node.
"""

import json
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

import operation_love.hub as hub
import operation_love.ranker as ranker
import operation_love.ranker.model as model_mod
from operation_love.hub import HubState, _Handler, _PAGE, _bind
from operation_love.hub.state import (
    _json_safe,
    validate_auto_review_answer,
    validate_auto_review_index,
    validate_auto_review_kind,
    validate_auto_review_profile_id,
)

NODE_BIN = shutil.which("node")
_LIVENESS_TIMEOUT_S = 15.0

PID = "0123456789abcdef0123456789abcdef"
PID2 = "fedcba9876543210fedcba9876543210"
PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"


# --- fakes -----------------------------------------------------------------------------------
def _pass(pid=PID, kind="random", **over):
    item = {
        "profile_id": pid,
        "app": "hinge",
        "run_id": "run-1",
        "created_at": 1_700_000_000.0,
        "score": 0.4123,
        "threshold": 0.4567,
        "photo_count": 3,
        "profile_name": "Avery",
        "kind": kind,
    }
    item.update(over)
    return item


class ReviewStore:
    """A store implementing the review surface; every call is logged in order."""

    def __init__(
        self, *, queue=None, photos=None, stats=None, answer=None, rows=None, undo=None
    ):
        self.undo = undo
        self.rows = list(
            rows if rows is not None else [(True, [0.1], ""), (False, [0.2], "")]
        )
        self.queue = dict(queue or {})
        self.photos = dict(photos or {})
        self.stats = stats if stats is not None else _stats()
        self.answer = answer
        self.calls = []
        self.closed = 0

    def next_auto_pass_for_review(self, *, kind, app=None):
        self.calls.append(("next", kind, app))
        return self.queue.get(kind)

    def auto_pass_photo(self, *, profile_id, index):
        self.calls.append(("photo", profile_id, index))
        return self.photos.get((profile_id, index))

    def record_auto_pass_review(self, *, profile_id, would_like):
        self.calls.append(("answer", profile_id, would_like))
        if self.answer is not None:
            return self.answer(profile_id, would_like)
        return {"kind": "random", "label_source": "auto_review_random"}

    def undo_latest_auto_review(self):
        self.calls.append(("undo",))
        if callable(self.undo):
            return self.undo()
        return self.undo

    def auto_review_stats(self, *, app=None):
        self.calls.append(("stats", app))
        return self.stats

    def load_label_rows(self):
        self.calls.append(("rows",))
        return list(self.rows)

    def flush(self):
        self.calls.append(("flush",))

    def close(self):
        self.closed += 1
        self.calls.append(("close",))


class LegacyStore:
    """A store from before the review archive: none of the review methods exist."""

    def __init__(self):
        self.closed = 0

    def load_labels(self):
        return []

    def close(self):
        self.closed += 1


def _stats(**over):
    stats = {
        "pending_random": 4,
        "pending_near_cut": 7,
        "random_reviewed": 0,
        "random_would_like": 0,
        "near_cut_reviewed": 0,
        "near_cut_would_like": 0,
        "auto_likes": 10,
        "auto_passes": 30,
    }
    stats.update(over)
    return stats


@pytest.fixture
def env(monkeypatch):
    """A HubState whose own stores come from a controllable factory (never a real backend)."""
    made = []
    state = {"factory": lambda: ReviewStore()}

    def make_store(cfg, ensure=True):
        store = state["factory"]()
        made.append((store, ensure))
        return store

    monkeypatch.setattr(hub.cfg_mod, "load", lambda path: SimpleNamespace(ranker=None))
    monkeypatch.setattr(ranker, "make_store", make_store)
    hub_state = HubState("config.yaml")
    return SimpleNamespace(
        hub=hub_state,
        made=made,
        use=lambda factory: state.__setitem__("factory", factory),
    )


def _go_live(hub_state, store=None, model=None):
    """Pretend a run is in flight; returns a release() that ends it."""
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    with hub_state._lock:
        hub_state._thread = thread
        hub_state._live_store = store
        hub_state._live_model = model

    def end():
        release.set()
        thread.join(_LIVENESS_TIMEOUT_S)

    return end


# --- validators --------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["random", "near_cut"])
def test_review_kind_accepts_only_the_two_pools(kind):
    assert validate_auto_review_kind(kind) == (True, kind, None)


@pytest.mark.parametrize(
    "bad", [None, "", "Random", "random ", "all", "near-cut", 1, ["random"], b"random"]
)
def test_review_kind_refuses_anything_else(bad):
    ok, value, error = validate_auto_review_kind(bad)
    assert (ok, value) == (False, None) and "kind" in error


def test_review_profile_id_is_exactly_32_lowercase_hex():
    assert validate_auto_review_profile_id(PID) == (True, PID, None)
    for bad in [
        "",
        PID[:-1],
        PID + "0",
        PID.upper(),
        PID + "\n",  # `$` would let a trailing newline through; fullmatch must not
        "../" * 8 + "etc/passwd",
        "0123456789abcdef/123456789abcdef",
        "0123456789abcdef0123456789abcde%",
        "0123456789abcdef0123456789abcdeg",
        None,
        123,
        [PID],
    ]:
        ok, value, error = validate_auto_review_profile_id(bad)
        assert (ok, value) == (False, None), bad
        assert "profile_id" in error


def test_review_index_is_a_small_non_negative_integer():
    assert validate_auto_review_index("0") == (True, 0, None)
    assert validate_auto_review_index("12") == (True, 12, None)
    assert validate_auto_review_index(7) == (True, 7, None)
    assert validate_auto_review_index("999") == (True, 999, None)
    for bad in [
        None,
        "",
        "-1",
        -1,
        "1.5",
        1.5,
        "abc",
        "٣",
        "00000",
        "1000",
        1000,
        True,
        False,
        [1],
    ]:
        ok, value, _ = validate_auto_review_index(bad)
        assert (ok, value) == (False, None), bad


def test_review_answer_requires_a_real_boolean_and_a_real_pass_id():
    assert validate_auto_review_answer({"profile_id": PID, "would_like": True}) == (
        True,
        (PID, True),
        None,
    )
    assert validate_auto_review_answer({"profile_id": PID, "would_like": False}) == (
        True,
        (PID, False),
        None,
    )
    # "false" is truthy, 1/0 are ints, null is missing: none may become a label.
    for would_like in ["true", "false", "yes", 1, 0, None, [], {}, 1.0]:
        ok, parsed, error = validate_auto_review_answer(
            {"profile_id": PID, "would_like": would_like}
        )
        assert (ok, parsed) == (False, None), would_like
        assert "boolean" in error
    ok, _, error = validate_auto_review_answer({"profile_id": PID})
    assert not ok and "boolean" in error
    for body in [
        {"would_like": True},
        {"profile_id": "nope", "would_like": True},
        {"profile_id": PID.upper(), "would_like": True},
        [],
        None,
        "x",
    ]:
        ok, parsed, _ = validate_auto_review_answer(body)
        assert (ok, parsed) == (False, None), body


def test_json_safe_makes_store_payloads_strict_json():
    class NumpyLike:
        def item(self):
            return 0.25

    payload = {
        "a": float("nan"),
        "b": [float("inf"), 1, True, None, "x"],
        "c": NumpyLike(),
        "d": (1.5,),
        3: object.__new__(type("X", (), {"__str__": lambda self: "X!"})),
    }
    safe = _json_safe(payload)
    assert safe["a"] is None and safe["b"] == [None, 1, True, None, "x"]
    assert safe["c"] == 0.25 and safe["d"] == [1.5] and safe["3"] == "X!"
    json.dumps(safe, allow_nan=False)  # would raise on NaN / Infinity


# --- /next -------------------------------------------------------------------------------------
def test_next_returns_the_requested_pool_first_and_closes_its_own_store(env):
    store = ReviewStore(queue={"random": _pass(), "near_cut": _pass(PID2, "near_cut")})
    env.use(lambda: store)

    code, payload = env.hub.auto_review_next("random")

    assert code == 200 and payload["available"] is True
    assert payload["item"]["profile_id"] == PID and payload["item"]["kind"] == "random"
    assert payload["fell_back"] is False and payload["requested_kind"] == "random"
    # Only the preferred pool was asked; nothing else was touched.
    assert [c for c in store.calls if c[0] == "next"] == [("next", "random", None)]
    # An idle hub opens its own READ-ONLY-setup store (no DDL) and closes it after use.
    assert env.made == [(store, False)]
    assert store.closed == 1


def test_next_falls_back_to_the_other_pool_in_one_store_session(env):
    store = ReviewStore(queue={"near_cut": _pass(PID2, "near_cut")})
    env.use(lambda: store)

    code, payload = env.hub.auto_review_next("random")

    assert code == 200
    assert (
        payload["item"]["profile_id"] == PID2 and payload["item"]["kind"] == "near_cut"
    )
    assert payload["fell_back"] is True and payload["requested_kind"] == "random"
    assert [c[1] for c in store.calls if c[0] == "next"] == ["random", "near_cut"]
    assert len(env.made) == 1


def test_next_with_both_pools_empty_is_an_available_empty_card(env):
    env.use(lambda: ReviewStore())

    code, payload = env.hub.auto_review_next("near_cut")

    assert code == 200
    assert payload == {
        "available": True,
        "requested_kind": "near_cut",
        "item": None,
        "fell_back": False,
    }


def test_next_whitelists_the_item_and_survives_non_finite_numbers(env):
    item = _pass(
        score=float("nan"), threshold=float("inf"), secret="leak", embedding=[1.0]
    )
    env.use(lambda: ReviewStore(queue={"random": item}))

    code, payload = env.hub.auto_review_next("random")

    assert code == 200
    assert set(payload["item"]) == {
        "profile_id",
        "kind",
        "app",
        "run_id",
        "created_at",
        "score",
        "threshold",
        "photo_count",
        "profile_name",
    }
    assert payload["item"]["score"] is None and payload["item"]["threshold"] is None
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    "bad",
    [
        "not a dict",
        _pass(pid="../../x"),
        _pass(photo_count=True),
        _pass(photo_count=-1),
        _pass(photo_count=5000),
        _pass(photo_count="3"),
    ],
)
def test_next_refuses_a_malformed_pass_from_the_store(env, bad):
    env.use(lambda: ReviewStore(queue={"random": bad}))

    code, payload = env.hub.auto_review_next("random")

    assert code == 502 and payload["available"] is False and payload["item"] is None
    assert "malformed" in payload["reason"]


def test_next_on_a_store_without_the_review_surface_is_unavailable_not_an_error(env):
    legacy = LegacyStore()
    env.use(lambda: legacy)

    code, payload = env.hub.auto_review_next("random")

    assert code == 200 and payload["available"] is False and payload["item"] is None
    assert "LegacyStore" in payload["reason"]
    assert "next_auto_pass_for_review" in payload["reason"]
    assert legacy.closed == 1


def test_next_reports_a_storage_failure_loudly(env):
    class Broken(ReviewStore):
        def next_auto_pass_for_review(self, *, kind, app=None):
            raise RuntimeError("table auto_passes not found")

    broken = Broken()
    env.use(lambda: broken)

    code, payload = env.hub.auto_review_next("random")

    assert code == 502 and payload["available"] is False
    assert "RuntimeError: table auto_passes not found" in payload["reason"]
    assert broken.closed == 1  # still closed on the failure path


# --- live store vs own store -------------------------------------------------------------------
def test_a_live_runs_store_is_used_instead_of_opening_another(env):
    live = ReviewStore(queue={"random": _pass()})
    end = _go_live(env.hub, store=live)
    try:
        code, payload = env.hub.auto_review_next("random")
    finally:
        end()

    assert code == 200 and payload["item"]["profile_id"] == PID
    assert env.made == []  # no second store was opened
    assert live.closed == 0  # the supervisor owns the live store's lifecycle


def test_a_read_retries_on_a_fresh_store_when_the_live_store_has_just_closed(env):
    class Closed(ReviewStore):
        def next_auto_pass_for_review(self, *, kind, app=None):
            raise RuntimeError("store is closed")

    fresh = ReviewStore(queue={"random": _pass()})
    env.use(lambda: fresh)
    end = _go_live(env.hub, store=Closed())
    try:
        code, payload = env.hub.auto_review_next("random")
    finally:
        end()

    assert code == 200 and payload["item"]["profile_id"] == PID
    assert len(env.made) == 1 and fresh.closed == 1


def test_an_answer_through_the_live_store_is_flushed_but_the_store_is_not_closed(env):
    live = ReviewStore()
    end = _go_live(env.hub, store=live)
    try:
        ok, _, _, code = env.hub.submit_auto_review_answer(
            {"profile_id": PID, "would_like": True}
        )
    finally:
        end()

    assert (ok, code) == (True, 200)
    # Durable before success is reported (a buffered row would die with the process), and the
    # store is left to the supervisor that owns it.
    assert [c[0] for c in live.calls] == ["answer", "flush"]
    assert live.closed == 0 and env.made == []


def test_a_write_never_retries_on_another_store_after_the_live_store_failed(env):
    def boom(profile_id, would_like):
        raise RuntimeError("live store closed mid-write")

    live = ReviewStore(answer=boom)
    fresh = ReviewStore()
    env.use(lambda: fresh)
    end = _go_live(env.hub, store=live)
    try:
        ok, result, msg, code = env.hub.submit_auto_review_answer(
            {"profile_id": PID, "would_like": True}
        )
    finally:
        end()

    assert (ok, code) == (False, 500) and "live store closed mid-write" in msg
    assert env.made == [] and fresh.calls == []  # a second attempt could double-record


def test_overlapping_requests_share_one_store_and_the_last_one_out_closes_it(env):
    """The browser fetches every photo in parallel; opening a BigQuery client per photo
    would cost more than the read.  The store must survive until the last user leaves."""
    store = ReviewStore()
    env.use(lambda: store)
    inside = threading.Barrier(3, timeout=_LIVENESS_TIMEOUT_S)
    leave = [threading.Event(), threading.Event()]
    seen = []

    def user(n):
        with env.hub._leased_review_store() as s:
            seen.append(s)
            inside.wait()
            leave[n].wait(_LIVENESS_TIMEOUT_S)

    threads = [threading.Thread(target=user, args=(n,)) for n in range(2)]
    for t in threads:
        t.start()
    inside.wait()
    assert store.closed == 0 and len(env.made) == 1

    # The FIRST user leaves while the second is still reading: the store must stay open.
    leave[0].set()
    threads[0].join(_LIVENESS_TIMEOUT_S)
    assert store.closed == 0 and env.hub._review_store is store

    leave[1].set()
    threads[1].join(_LIVENESS_TIMEOUT_S)

    assert seen == [store, store]
    assert store.closed == 1  # exactly once, by the last user out
    assert env.hub._review_store is None

    # Sequential use afterwards opens a fresh store (nothing is held open between requests).
    with env.hub._leased_review_store():
        pass
    assert len(env.made) == 2


def test_a_failed_store_open_does_not_leak_a_user_count(env):
    attempts = []

    def factory():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("no credentials")
        return ReviewStore()

    env.use(factory)

    code, payload = env.hub.auto_review_next("random")
    assert code == 502 and "no credentials" in payload["reason"]

    code, payload = env.hub.auto_review_next("random")  # recovers: count was not leaked
    assert code == 200 and env.hub._review_store is None
    assert env.hub._review_store_users == 0


# --- /image ------------------------------------------------------------------------------------
def test_image_returns_the_bytes_with_a_sniffed_media_type(env):
    jpeg = b"\xff\xd8\xff\xe0" + b"jpeg-body"
    webp = b"RIFF\x10\x00\x00\x00WEBPVP8 "
    store = ReviewStore(photos={(PID, 0): PNG, (PID, 1): jpeg, (PID, 2): webp})
    env.use(lambda: store)

    assert env.hub.auto_review_image(PID, 0) == (200, PNG, "image/png")
    assert env.hub.auto_review_image(PID, 1) == (200, jpeg, "image/jpeg")
    assert env.hub.auto_review_image(PID, 2) == (200, webp, "image/webp")
    assert ("photo", PID, 1) in store.calls


def test_image_that_is_not_an_image_is_refused_not_served(env):
    env.use(lambda: ReviewStore(photos={(PID, 0): b"<html><script>alert(1)</script>"}))

    code, data, message = env.hub.auto_review_image(PID, 0)

    assert code == 502 and data is None and "not a recognised image" in message


def test_image_missing_unknown_or_unsupported(env):
    env.use(lambda: ReviewStore())
    assert env.hub.auto_review_image(PID, 5)[0] == 404

    class Raising(ReviewStore):
        def auto_pass_photo(self, *, profile_id, index):
            raise ValueError("unknown auto pass")

    env.use(lambda: Raising())
    code, data, message = env.hub.auto_review_image(PID, 0)
    assert (code, data, message) == (404, None, "unknown auto pass")

    env.use(lambda: LegacyStore())
    assert env.hub.auto_review_image(PID, 0)[0] == 501


# --- /answer -----------------------------------------------------------------------------------
def test_answer_records_flushes_then_reports_and_closes(env):
    store = ReviewStore()
    env.use(lambda: store)

    ok, result, msg, code = env.hub.submit_auto_review_answer(
        {"profile_id": PID, "would_like": True}
    )

    assert (ok, code, msg) == (True, 200, "recorded")
    assert result == {"kind": "random", "label_source": "auto_review_random"}
    # The label is made durable BEFORE success is reported; close follows.
    assert [c[0] for c in store.calls] == ["answer", "flush", "close"]
    assert store.calls[0] == ("answer", PID, True)
    # The answer moved the live evidence: the cached calibration is stale.
    assert env.hub._calibration_dirty is True


def test_answer_can_record_a_correct_pass(env):
    store = ReviewStore()
    env.use(lambda: store)

    ok, _, _, _ = env.hub.submit_auto_review_answer(
        {"profile_id": PID, "would_like": False}
    )

    assert ok and store.calls[0] == ("answer", PID, False)


@pytest.mark.parametrize(
    "body",
    [
        {"profile_id": PID, "would_like": "true"},
        {"profile_id": PID, "would_like": 1},
        {"profile_id": PID},
        {"profile_id": "../../etc", "would_like": True},
        {"would_like": True},
        [],
    ],
)
def test_an_invalid_answer_never_reaches_a_store(env, body):
    env.use(
        lambda: pytest.fail("a store must not even be opened for an invalid answer")
    )

    ok, result, msg, code = env.hub.submit_auto_review_answer(body)

    assert (ok, result, code) == (False, None, 400) and msg
    assert env.made == []


def test_answering_an_unknown_or_already_answered_pass_is_a_retiring_conflict(env):
    def refuse(profile_id, would_like):
        raise ValueError("auto pass already reviewed")

    env.use(lambda: ReviewStore(answer=refuse))

    ok, result, msg, code = env.hub.submit_auto_review_answer(
        {"profile_id": PID, "would_like": True}
    )

    assert (ok, code) == (False, 409)
    assert result == {
        "retired": True
    }  # the page moves on instead of retrying this card
    assert "already reviewed" in msg
    assert env.hub._calibration_dirty is False  # nothing changed


def test_answer_storage_failure_is_a_500_with_the_cause(env):
    def boom(profile_id, would_like):
        raise RuntimeError("BigQuery 503")

    env.use(lambda: ReviewStore(answer=boom))

    ok, result, msg, code = env.hub.submit_auto_review_answer(
        {"profile_id": PID, "would_like": True}
    )

    assert (ok, result, code) == (
        False,
        None,
        500,
    ) and "RuntimeError: BigQuery 503" in msg


def test_answer_on_a_store_without_the_surface_is_501(env):
    env.use(lambda: LegacyStore())

    ok, _, msg, code = env.hub.submit_auto_review_answer(
        {"profile_id": PID, "would_like": True}
    )

    assert (ok, code) == (False, 501) and "record_auto_pass_review" in msg


def test_answer_is_refused_while_training_data_is_being_rewritten(env):
    env.use(lambda: pytest.fail("must not touch the store"))
    env.hub._training_data_mutating = True

    ok, _, msg, code = env.hub.submit_auto_review_answer(
        {"profile_id": PID, "would_like": True}
    )

    assert (ok, code) == (False, 409) and "training data" in msg


# --- /undo -------------------------------------------------------------------------------------
_UNDONE = {
    "profile_id": PID,
    "profile_name": "Avery",
    "kind": "near_cut",
    "would_like": True,
}


def test_undo_deletes_the_newest_answer_flushes_then_reports_and_closes(env):
    store = ReviewStore(undo=dict(_UNDONE))
    env.use(lambda: store)

    ok, result, msg, code = env.hub.undo_auto_review_answer()

    assert (ok, code, msg) == (True, 200, "undone")
    assert result == _UNDONE
    # Durable BEFORE success is reported, exactly like an answer; close follows.
    assert [c[0] for c in store.calls] == ["undo", "flush", "close"]
    # The undone answer left the live evidence and the training set: the calibration is stale.
    assert env.hub._calibration_dirty is True


def test_undo_with_nothing_to_undo_is_a_404_and_changes_nothing(env):
    env.use(lambda: ReviewStore(undo=None))

    ok, result, msg, code = env.hub.undo_auto_review_answer()

    assert (ok, result, code) == (False, None, 404) and "no review answer" in msg
    assert env.hub._calibration_dirty is False


def test_undo_goes_through_the_live_runs_store(env):
    live = ReviewStore(undo=dict(_UNDONE))
    env.use(lambda: pytest.fail("a live run's store must be used, not a second one"))
    end = _go_live(env.hub, store=live)
    try:
        ok, _, _, code = env.hub.undo_auto_review_answer()
    finally:
        end()

    assert (ok, code) == (True, 200)
    assert [c[0] for c in live.calls] == ["undo", "flush"]  # flushed, never closed
    assert live.closed == 0


def test_undo_on_a_store_without_the_surface_is_501(env):
    env.use(lambda: LegacyStore())

    ok, _, msg, code = env.hub.undo_auto_review_answer()

    assert (ok, code) == (False, 501) and "undo_latest_auto_review" in msg


def test_undo_storage_failure_is_a_500_with_the_cause_and_is_never_retried(env):
    attempts = []

    def boom():
        attempts.append(1)
        raise RuntimeError("BigQuery refused: streaming buffer")

    live = ReviewStore(undo=boom)
    env.use(lambda: pytest.fail("a failed write must not be retried on another store"))
    end = _go_live(env.hub, store=live)
    try:
        ok, result, msg, code = env.hub.undo_auto_review_answer()
    finally:
        end()

    assert (ok, result, code) == (False, None, 500)
    assert "RuntimeError: BigQuery refused" in msg
    assert attempts == [1]
    assert env.hub._calibration_dirty is False


def test_undo_is_refused_while_training_data_is_being_rewritten(env):
    env.use(lambda: pytest.fail("must not touch the store"))
    env.hub._training_data_mutating = True

    ok, _, msg, code = env.hub.undo_auto_review_answer()

    assert (ok, code) == (False, 409) and "training data" in msg


def test_concurrent_answers_for_one_pass_record_exactly_one_label(env):
    answered = set()
    entered = threading.Event()

    def answer(profile_id, would_like):
        if profile_id in answered:
            raise ValueError("already reviewed")
        entered.set()
        time.sleep(0.05)  # widen the race window
        answered.add(profile_id)
        return {"kind": "random", "label_source": "auto_review_random"}

    env.use(lambda: ReviewStore(answer=answer))
    results = []

    def submit():
        results.append(
            env.hub.submit_auto_review_answer({"profile_id": PID, "would_like": True})
        )

    threads = [threading.Thread(target=submit) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(_LIVENESS_TIMEOUT_S)

    assert sorted(code for _, _, _, code in results) == [200, 409, 409, 409]


# --- /stats + calibration ----------------------------------------------------------------------
CAL = {
    "recall_target": 0.95,
    "effective_target": 0.96,
    "threshold": 0.4712,
    "expected_false_pass_rate": 0.05,
    "base_rate": 0.4,
    "calibration_labels": 300,
    "review_labels": 12,
    "live": {"random_reviewed": 40, "extra": 0.01},
    "max_threshold_lift": 0.012,
}


def test_stats_with_the_live_runs_model_compares_measured_to_expected(env):
    env.use(lambda: ReviewStore(stats=_stats(random_reviewed=40, random_would_like=10)))
    model = SimpleNamespace(calibration=CAL, recall_target=0.95)
    end = _go_live(
        env.hub,
        store=ReviewStore(stats=_stats(random_reviewed=40, random_would_like=10)),
        model=model,
    )
    try:
        code, payload = env.hub.auto_review_stats_snapshot()
    finally:
        end()

    assert code == 200 and payload["available"] is True
    assert payload["calibration"]["threshold"] == 0.4712
    assert payload["calibration_source"] == "live_run"
    assert payload["calibration_pending"] is False
    summary = payload["summary"]
    assert summary["random_reviewed"] == 40 and summary["random_would_like"] == 10
    assert summary["measured_false_pass_rate"] == pytest.approx(0.25)
    assert summary["expected_false_pass_rate"] == 0.05
    # 10/40 with a 95% lower bound near 0.15 is clearly above the predicted 5%.
    assert summary["observed_false_pass_lower"] > 0.05
    assert summary["verdict"] == "above"
    assert payload["store"]["pending_near_cut"] == 7
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("reviewed", "would_like", "expected", "verdict"),
    [
        (9, 9, 0.05, "insufficient"),  # below the 10-answer floor, however bad it looks
        (0, 0, 0.05, "insufficient"),
        (40, 40, None, "unknown"),  # no prediction to compare with
        (40, 1, 0.05, "within"),
        (200, 14, 0.05, "within"),  # 7% observed but the lower bound is under 5%
        (200, 30, 0.05, "above"),
    ],
)
def test_summary_verdict(reviewed, would_like, expected, verdict):
    stats = _stats(random_reviewed=reviewed, random_would_like=would_like)
    cal = None if expected is None else {"expected_false_pass_rate": expected}

    summary = HubState._review_summary(stats, cal)

    assert summary["verdict"] == verdict
    assert summary["random_reviewed"] == reviewed
    if reviewed == 0:
        assert summary["measured_false_pass_rate"] is None


def test_summary_tolerates_garbage_counts():
    summary = HubState._review_summary(
        {"random_reviewed": "x", "random_would_like": True},
        {"expected_false_pass_rate": True},
    )
    assert summary["random_reviewed"] == 0 and summary["verdict"] == "insufficient"
    assert summary["expected_false_pass_rate"] is None


def test_stats_on_a_store_without_the_surface_say_unavailable(env):
    env.use(lambda: LegacyStore())

    code, payload = env.hub.auto_review_stats_snapshot()

    assert code == 200 and payload["available"] is False
    assert "auto_review_stats" in payload["reason"]
    assert payload["summary"] is None and payload["store"] is None


def test_stats_storage_failure_is_a_502_with_the_cause(env):
    class Broken(ReviewStore):
        def auto_review_stats(self, *, app=None):
            raise RuntimeError("quota")

    env.use(lambda: Broken())

    code, payload = env.hub.auto_review_stats_snapshot()

    assert (
        code == 502 and payload["available"] is False and "quota" in payload["reason"]
    )


def test_an_idle_hub_computes_the_calibration_in_the_background_then_serves_it(
    env, monkeypatch
):
    env.use(lambda: ReviewStore(stats=_stats(random_reviewed=20, random_would_like=1)))
    seen = {}

    class FakeModel:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            self.not_ready_reason = None

        def train(self, samples, *, live_review=None):
            seen["samples"] = samples
            seen["live_review"] = live_review
            return True

        @property
        def calibration(self):
            return dict(CAL)

    monkeypatch.setattr(model_mod, "PreferenceModel", FakeModel)
    ranker_cfg = SimpleNamespace(
        min_labels_to_engage=15,
        like_threshold=0.5,
        min_per_class=4,
        like_recall_target=0.95,
        contextual_recall_budget=0.03,
    )
    monkeypatch.setattr(
        hub.cfg_mod, "load", lambda path: SimpleNamespace(ranker=ranker_cfg)
    )

    code, first = env.hub.auto_review_stats_snapshot()
    assert code == 200
    # Nothing to show yet, but the card is told to look again.
    assert first["calibration"] is None and first["calibration_pending"] is True

    deadline = time.monotonic() + _LIVENESS_TIMEOUT_S
    while env.hub._calibration_refreshing and time.monotonic() < deadline:
        time.sleep(0.01)
    code, second = env.hub.auto_review_stats_snapshot()

    assert second["calibration"]["threshold"] == 0.4712
    assert (
        second["calibration_source"] == "computed"
        and second["calibration_pending"] is False
    )
    # Built exactly as a run builds it: the configured recall target and budget, trained on the
    # store's rows together with the live review stats.
    assert seen["init"] == {
        "min_labels": 15,
        "threshold": 0.5,
        "min_per_class": 4,
        "recall_target": 0.95,
        "contextual_recall_budget": 0.03,
    }
    assert seen["live_review"]["random_reviewed"] == 20
    assert seen["samples"] == [(True, [0.1], ""), (False, [0.2], "")]
    # Fresh: asking again does not recompute.
    made_before = len(env.made)
    env.hub.auto_review_stats_snapshot()
    time.sleep(0.05)
    assert len([m for m in env.made[made_before:]]) <= 1  # only the stats read itself


def test_an_answer_makes_the_cached_calibration_stale(env, monkeypatch):
    env.use(lambda: ReviewStore())
    calls = []
    monkeypatch.setattr(
        HubState, "_compute_review_calibration", lambda self: calls.append(1)
    )
    with env.hub._lock:
        env.hub._calibration_cache = dict(CAL)
        env.hub._calibration_attempt_at = time.time()

    env.hub._review_calibration()  # fresh: no recompute
    time.sleep(0.05)
    assert calls == []

    env.hub.submit_auto_review_answer({"profile_id": PID, "would_like": False})
    cached, source, pending, error = env.hub._review_calibration()

    deadline = time.monotonic() + _LIVENESS_TIMEOUT_S
    while env.hub._calibration_refreshing and time.monotonic() < deadline:
        time.sleep(0.01)
    # The stale figures are still served while the refresh runs.
    assert source == "computed" and cached["threshold"] == 0.4712
    assert calls == [1]


def test_a_failed_calibration_is_reported_and_not_retried_every_poll(env, monkeypatch):
    env.use(lambda: ReviewStore())
    attempts = []

    class Failing:
        def __init__(self, **kwargs):
            attempts.append(1)
            raise ValueError("min_labels must be a positive integer")

    monkeypatch.setattr(model_mod, "PreferenceModel", Failing)
    ranker_cfg = SimpleNamespace(
        min_labels_to_engage=0,
        like_threshold=0.5,
        min_per_class=5,
        like_recall_target=0.95,
    )
    monkeypatch.setattr(
        hub.cfg_mod, "load", lambda path: SimpleNamespace(ranker=ranker_cfg)
    )

    env.hub.auto_review_stats_snapshot()
    deadline = time.monotonic() + _LIVENESS_TIMEOUT_S
    while env.hub._calibration_refreshing and time.monotonic() < deadline:
        time.sleep(0.01)
    _, payload = env.hub.auto_review_stats_snapshot()
    _, again = env.hub.auto_review_stats_snapshot()

    assert payload["calibration"] is None and payload["calibration_pending"] is False
    assert "ValueError: min_labels" in payload["calibration_error"]
    assert again["calibration_error"] == payload["calibration_error"]
    assert attempts == [1]  # one expensive attempt, not one per poll


def test_a_legacy_fixed_threshold_run_is_explained_not_recomputed(env):
    env.use(lambda: ReviewStore())
    model = SimpleNamespace(calibration=None, recall_target=None)
    end = _go_live(env.hub, store=ReviewStore(), model=model)
    try:
        _, payload = env.hub.auto_review_stats_snapshot()
    finally:
        end()

    assert payload["calibration"] is None and payload["calibration_pending"] is False
    assert "fixed threshold" in payload["calibration_error"]
    assert env.hub._calibration_refreshing is False


def test_a_finished_runs_calibration_is_frozen_as_plain_data(env):
    model = SimpleNamespace(calibration=dict(CAL), recall_target=0.95)
    with env.hub._lock:
        env.hub._live_model = model

    env.hub._freeze_completed_calibration()
    with env.hub._lock:
        env.hub._live_model = None
        env.hub._calibration_attempt_at = (
            time.time()
        )  # suppress the background recompute

    cached, source, pending, error = env.hub._review_calibration()
    assert source == "last_run" and cached["threshold"] == 0.4712
    assert cached is not model.calibration  # a copy, not a view of the model


def test_binding_a_worker_captures_the_shared_model(env):
    model = object()
    worker = SimpleNamespace(
        run_id="run-1", app="hinge", decider=SimpleNamespace(model=model)
    )

    env.hub._bind_training_worker(worker)

    assert env.hub._live_model is model
    # A worker whose decider has no model (a fake / third-party Decider) must not clobber it.
    env.hub._bind_training_worker(
        SimpleNamespace(run_id="run-1", app="bumble", decider=object())
    )
    assert env.hub._live_model is model


# --- the model-quality card must not judge the ranker on its own past passes ----------------------
def test_eval_labels_exclude_answers_to_archived_passes():
    class Store:
        def load_label_rows(self):
            return [
                (True, [1.0], ""),
                (False, [2.0], "manual"),
                (True, [3.0], None),
                (True, [4.0], "auto_review_random"),
                (False, [5.0], "auto_review_near_cut"),
                (True, [6.0]),  # a double that still returns pairs
            ]

        def load_labels(self):
            raise AssertionError("must prefer load_label_rows")

    assert HubState._eval_label_pairs(Store()) == [
        (True, [1.0]),
        (False, [2.0]),
        (True, [3.0]),
        (True, [6.0]),
    ]


def test_eval_labels_fall_back_to_load_labels_for_an_older_store():
    class Old:
        def load_labels(self):
            return [(True, [1.0])]

    assert HubState._eval_label_pairs(Old()) == [(True, [1.0])]


def test_the_eval_card_judges_only_manual_labels(env, monkeypatch):
    import operation_love.ranker.evaluate as eval_mod

    captured = {}

    def fake_evaluate(samples, **kwargs):
        captured["samples"] = list(samples)
        return {"status": "insufficient_data", "labels": len(samples)}

    class Rows(ReviewStore):
        def load_label_rows(self):
            return [
                (True, [1.0], ""),
                (False, [2.0], "auto_review_near_cut"),
                (True, [3.0], "auto_review_random"),
            ]

    monkeypatch.setattr(eval_mod, "evaluate", fake_evaluate)
    env.use(lambda: Rows())

    snapshot = env.hub.eval_snapshot()

    assert captured["samples"] == [(True, [1.0])]
    assert snapshot["labels"] == 1


# --- the model-quality card must evaluate at the threshold the bot deploys ---------------------
def _eval_cfg(monkeypatch, **ranker_fields):
    ranker_cfg = SimpleNamespace(like_threshold=0.5, like_recall_target=None)
    for key, value in ranker_fields.items():
        setattr(ranker_cfg, key, value)
    monkeypatch.setattr(
        hub.cfg_mod, "load", lambda path: SimpleNamespace(ranker=ranker_cfg)
    )


def _capture_evaluate(monkeypatch, result):
    import operation_love.ranker.evaluate as eval_mod

    seen = {}

    def fake_evaluate(samples, **kwargs):
        seen.update(kwargs)
        return dict(result, labels=len(samples))

    monkeypatch.setattr(eval_mod, "evaluate", fake_evaluate)
    return seen


_OK = {"status": "ok", "accepted_recall": 0.96, "false_dislike_rate": 0.04}


class _CalibratedModel:
    recall_target = 0.95
    not_ready_reason = None

    def __init__(self, threshold):
        self.calibration = (
            None
            if threshold is None
            else {"threshold": threshold, "recall_target": 0.95}
        )


def test_eval_without_a_recall_target_keeps_the_configured_threshold(env, monkeypatch):
    _eval_cfg(monkeypatch, like_threshold=0.62)
    seen = _capture_evaluate(monkeypatch, _OK)
    env.use(lambda: ReviewStore())

    snapshot = env.hub.eval_snapshot()

    assert seen["like_threshold"] == 0.62
    assert snapshot["threshold_source"] == "configured"
    assert snapshot["recall_target"] is None and snapshot["status"] == "ok"


def test_eval_with_a_recall_target_uses_the_running_models_calibrated_cutoff(
    env, monkeypatch
):
    _eval_cfg(monkeypatch, like_threshold=0.5, like_recall_target=0.95)
    seen = _capture_evaluate(monkeypatch, _OK)
    store = ReviewStore()
    env.use(lambda: store)
    end = _go_live(env.hub, store=store, model=_CalibratedModel(0.4812))
    try:
        snapshot = env.hub.eval_snapshot()
    finally:
        end()

    assert seen["like_threshold"] == pytest.approx(0.4812)
    assert snapshot["threshold_source"] == "calibrated"
    assert snapshot["recall_target"] == 0.95 and snapshot["status"] == "ok"


def test_eval_uses_the_last_runs_cutoff_when_idle(env, monkeypatch):
    _eval_cfg(monkeypatch, like_recall_target=0.95)
    seen = _capture_evaluate(monkeypatch, _OK)
    env.use(lambda: ReviewStore())
    env.hub._last_calibration = {"threshold": 0.4444, "recall_target": 0.95}

    snapshot = env.hub.eval_snapshot()

    assert seen["like_threshold"] == pytest.approx(0.4444)
    assert snapshot["threshold_source"] == "calibrated"
    assert snapshot["recall_target"] == 0.95


def test_eval_ignores_a_cached_cutoff_made_for_a_different_target(env, monkeypatch):
    """The owner edited like_recall_target (0.95 -> 0.80) with the hub still open.  The last
    run's cutoff was calibrated for 0.95: showing it as "the cutoff calibrated to keep >=80%"
    (and grading recall against it) describes a bot that does not exist."""
    _eval_cfg(monkeypatch, like_recall_target=0.80)
    seen = _capture_evaluate(monkeypatch, _OK)
    env.use(lambda: ReviewStore())  # too few labels for an out-of-fold estimate
    env.hub._last_calibration = {"threshold": 0.4444, "recall_target": 0.95}
    env.hub._calibration_cache = {"threshold": 0.4555, "recall_target": 0.90}

    snapshot = env.hub.eval_snapshot()

    assert snapshot["threshold_source"] == "uncalibrated"
    assert seen["like_threshold"] == 0.5  # neither stale cutoff was substituted
    assert snapshot["status"] == "uncalibrated" and snapshot["accepted_recall"] is None


def test_eval_uses_a_cached_cutoff_only_for_the_current_target(env, monkeypatch):
    _eval_cfg(monkeypatch, like_recall_target=0.80)
    seen = _capture_evaluate(monkeypatch, _OK)
    env.use(lambda: ReviewStore())
    env.hub._calibration_cache = {"threshold": 0.4555, "recall_target": 0.90}  # stale
    env.hub._last_calibration = {"threshold": 0.4111, "recall_target": 0.80}  # current

    snapshot = env.hub.eval_snapshot()

    assert snapshot["threshold_source"] == "calibrated"
    assert seen["like_threshold"] == pytest.approx(0.4111)
    assert snapshot["recall_target"] == 0.80


def test_eval_ignores_a_cached_cutoff_that_does_not_say_what_it_was_made_for(
    env, monkeypatch
):
    _eval_cfg(monkeypatch, like_recall_target=0.95)
    _capture_evaluate(monkeypatch, _OK)
    env.use(lambda: ReviewStore())
    env.hub._last_calibration = {"threshold": 0.4444}  # no recall_target: whose cutoff?

    assert env.hub.eval_snapshot()["threshold_source"] == "uncalibrated"


def test_eval_shows_a_running_bots_cutoff_with_the_target_it_was_built_for(
    env, monkeypatch
):
    # The bot started under 0.80; config now says 0.95.  The bot is what it is, and the card
    # describes it: its cutoff, labelled with ITS target (not the edited config's).
    _eval_cfg(monkeypatch, like_recall_target=0.95)
    seen = _capture_evaluate(monkeypatch, _OK)
    store = ReviewStore()
    env.use(lambda: store)

    class _Model:
        recall_target = 0.80
        not_ready_reason = None
        calibration = {"threshold": 0.4812, "recall_target": 0.80}

    end = _go_live(env.hub, store=store, model=_Model())
    try:
        snapshot = env.hub.eval_snapshot()
    finally:
        end()

    assert seen["like_threshold"] == pytest.approx(0.4812)
    assert (
        snapshot["threshold_source"] == "calibrated"
        and snapshot["recall_target"] == 0.80
    )
    assert seen["recall_target"] == 0.80


def test_eval_with_a_recall_target_asks_for_the_nested_held_out_test(env, monkeypatch):
    _eval_cfg(monkeypatch, like_recall_target=0.95)
    seen = _capture_evaluate(monkeypatch, _OK)
    env.use(lambda: ReviewStore())
    env.hub._last_calibration = {"threshold": 0.4444, "recall_target": 0.95}
    env.hub.eval_snapshot()
    assert (
        seen["recall_target"] == 0.95
    )  # else the card grades the labels the cutoff came from

    # No target: the legacy fixed-threshold evaluation, byte for byte (no new keyword at all).
    _eval_cfg(monkeypatch, like_threshold=0.62)
    seen = _capture_evaluate(monkeypatch, _OK)
    HubState("config.yaml").eval_snapshot()
    assert "recall_target" not in seen

    # An uncalibrated placeholder is refused outright and never evaluated as a nested test.
    _eval_cfg(monkeypatch, like_recall_target=0.95)
    seen = _capture_evaluate(monkeypatch, _OK)
    stale = HubState("config.yaml")
    stale._last_calibration = {"threshold": 0.4, "recall_target": 0.5}
    assert stale.eval_snapshot()["status"] == "uncalibrated"
    assert "recall_target" not in seen


def test_eval_never_substitutes_an_old_cutoff_for_a_running_models_missing_one(
    env, monkeypatch
):
    # The running bot has a recall-target model with no calibration: its threshold is not the
    # cached one, so the card must not present the cached one as "what the bot uses".
    _eval_cfg(monkeypatch, like_recall_target=0.95)
    seen = _capture_evaluate(monkeypatch, _OK)
    store = ReviewStore()
    env.use(lambda: store)
    env.hub._calibration_cache = {"threshold": 0.4444, "recall_target": 0.95}
    end = _go_live(env.hub, store=store, model=_CalibratedModel(None))
    try:
        snapshot = env.hub.eval_snapshot()
    finally:
        end()

    # Too few labels for an out-of-fold estimate: the placeholder is refused outright.
    assert seen["like_threshold"] == 0.5
    assert snapshot["threshold_source"] == "uncalibrated"
    assert snapshot["status"] == "uncalibrated"
    assert snapshot["accepted_recall"] is None
    assert snapshot["false_dislike_rate"] is None
    assert "no calibrated threshold" in snapshot["message"]


def test_eval_estimates_the_cutoff_from_its_own_out_of_fold_scores_when_idle(
    env, monkeypatch
):
    import numpy as np

    rng = np.random.default_rng(7)
    rows = []
    for i in range(80):
        vec = rng.normal(size=1280)
        liked = bool(i % 2)
        vec[600] += 1.5 if liked else -1.5  # a learnable, identity-independent signal
        rows.append((liked, vec.tolist(), ""))
    _eval_cfg(monkeypatch, like_recall_target=0.9)
    seen = _capture_evaluate(monkeypatch, _OK)
    env.use(lambda: ReviewStore(rows=rows))

    snapshot = env.hub.eval_snapshot()

    assert snapshot["threshold_source"] == "oof_estimate"
    assert 0.0 < seen["like_threshold"] < 1.0 and seen["like_threshold"] != 0.5
    assert snapshot["status"] == "ok"


def test_oof_estimate_refuses_a_target_the_manual_likes_cannot_measure():
    """The model is NOT ready on fewer manual likes than the target can be measured from, so
    an estimate here would describe a cutoff the bot never uses."""
    import numpy as np

    rng = np.random.default_rng(3)
    rows = []
    for i in range(70):
        vec = rng.normal(size=1280)
        liked = (
            i < 12
        )  # 12 likes: clears min_per_class, cannot measure a 0.95 recall (needs 19)
        vec[600] += 1.5 if liked else -1.5
        rows.append((liked, vec.tolist(), ""))
    samples = [(liked, vec) for liked, vec, _ in rows]
    assert HubState._oof_recall_threshold(samples, 0.95) is None
    assert HubState._oof_recall_threshold(samples, 0.90) is not None  # 12 >= 9


def test_oof_recall_threshold_gives_up_quietly_on_unusable_labels():
    only_likes = [(True, [float(i), 1.0]) for i in range(20)]
    assert HubState._oof_recall_threshold(only_likes, 0.95) is None
    assert HubState._oof_recall_threshold([(True, [1.0]), (False, [0.0])], 0.95) is None
    # Ragged embeddings are a data fault: reported, not raised into the card.
    ragged = [(bool(i % 2), [1.0] * (2 + i % 2)) for i in range(20)]
    assert HubState._oof_recall_threshold(ragged, 0.95) is None


def test_the_page_judges_recall_against_the_owners_own_target():
    # The card reads the target from the response instead of hard-coding 95%.
    assert "e.recall_target" in _PAGE
    assert "percent(TARGET_RECALL)" in _PAGE
    assert "Target ≥95%" not in _PAGE


# --- the model-quality card's badge ----------------------------------------------------------
_RENDER_EVAL_HARNESS = r"""
const els = {};
function mk(id){ return els[id] = els[id] || {innerHTML:'', textContent:'', className:'', style:{}}; }
const $ = sel => mk(sel);
"""


def _render_eval(evaluation):
    script = (
        _RENDER_EVAL_HARNESS
        + _extract("renderEval")
        + f"\nrenderEval({json.dumps(evaluation)});\n"
        + "console.log(JSON.stringify({badge: els['#evalhint'].textContent,"
        + " cls: els['#evalhint'].className, body: els['#evalbody'].innerHTML}));"
    )
    return _node(script)


def _ok_eval(**over):
    base = {
        "status": "ok",
        "accepted_recall": 0.96,
        "false_dislike_rate": 0.04,
        "identities": 600,
        "like_threshold": 0.47,
        "folds": 5,
        "confusion": {
            "true_accepted": 96,
            "false_dislikes": 4,
            "false_likes": 60,
            "true_dislikes": 40,
        },
        "threshold_source": "calibrated",
    }
    base.update(over)
    return base


_TARGETED = {
    "recall_target": 0.95,
    "accepted_recall_upper": 0.985,
    "dislikes_passed": 0.40,
    "dislikes_passed_lower": 0.32,
    "chance_pass_rate": 0.05,
    "discriminates": True,
}


def test_the_badge_passes_a_ranker_that_holds_recall_and_beats_chance():
    out = _render_eval(_ok_eval(**_TARGETED))
    assert out["badge"] == "BASELINE PASS" and "ready" in out["cls"]
    assert "never saw" in out["body"] and "Better than chance" in out["body"]


def test_the_badge_does_not_pass_a_ranker_that_only_meets_the_recall_target():
    """The original defect: under a recall target, recall at the cutoff is green for ANY ranker
    (liking everyone meets it), so a ranker with no signal showed 'BASELINE PASS' once enough
    identities were tested.  The quality check is what makes the badge depend on the model."""
    useless = dict(
        _TARGETED,
        discriminates=False,
        dislikes_passed=0.04,
        dislikes_passed_lower=0.02,
    )
    out = _render_eval(_ok_eval(accepted_recall=0.953, **useless))
    assert out["badge"] == "NOT READY" and "not-ready" in out["cls"]
    assert "More baseline evidence needed" in out["body"]


def test_the_badge_treats_an_absent_quality_verdict_as_not_passed():
    # A response that predates the quality check (or lost it) must not pass by omission.
    without = {k: v for k, v in _TARGETED.items() if k != "discriminates"}
    assert _render_eval(_ok_eval(**without))["badge"] == "NOT READY"
    assert (
        _render_eval(_ok_eval(**dict(_TARGETED, discriminates=None)))["badge"]
        == "NOT READY"
    )


def test_the_badge_fails_recall_only_when_the_upper_bound_is_under_the_target():
    # A correctly calibrated 0.95 target measured at 94% from ~150 positives must not flicker
    # to NOT READY: its 95% upper bound (97%) is above the target.
    near = _render_eval(
        _ok_eval(**dict(_TARGETED, accepted_recall_upper=0.971), accepted_recall=0.94)
    )
    assert near["badge"] == "BASELINE PASS"
    # A recall whose upper bound is itself below the target is a real miss.
    miss = _render_eval(
        _ok_eval(**dict(_TARGETED, accepted_recall_upper=0.93), accepted_recall=0.90)
    )
    assert miss["badge"] == "NOT READY"


def test_the_badge_still_needs_five_hundred_identities():
    assert _render_eval(_ok_eval(identities=499, **_TARGETED))["badge"] == "NOT READY"
    assert (
        _render_eval(_ok_eval(identities=500, **_TARGETED))["badge"] == "BASELINE PASS"
    )


def test_the_legacy_fixed_threshold_card_is_unchanged():
    # No recall_target in the response: the original rule (point-estimate recall >= 95% and 500
    # identities), no quality row, and no 'never saw' wording.
    legacy = _render_eval(_ok_eval(threshold_source="configured"))
    assert legacy["badge"] == "BASELINE PASS"
    assert (
        "Better than chance" not in legacy["body"] and "never saw" not in legacy["body"]
    )
    below = _render_eval(_ok_eval(accepted_recall=0.94, threshold_source="configured"))
    assert below["badge"] == "NOT READY"


# --- the real SQLite store, end to end -------------------------------------------------------
def test_review_flow_against_the_real_sqlite_store(tmp_path, monkeypatch):
    from operation_love.ranker.store import SQLiteStore

    db = tmp_path / "love.db"
    seed = SQLiteStore(db)
    try:
        for pid, random_sample, score in ((PID, True, 0.30), (PID2, False, 0.48)):
            assert seed.record_auto_pass(
                run_id="run-1",
                app="hinge",
                profile_id=pid,
                score=score,
                threshold=0.5,
                embedding=[0.1, 0.2, 0.3],
                photos=[PNG, PNG + b"-2"],
                profile_name="Avery" if random_sample else "Blake",
                random_sample=random_sample,
            )
        seed.add_label("run-0", "hinge", True, [0.9, 0.8, 0.7], source="manual")
    finally:
        seed.close()

    opened = []

    def make_store(cfg, ensure=True):
        store = SQLiteStore(db)
        opened.append(store)
        return store

    monkeypatch.setattr(hub.cfg_mod, "load", lambda path: SimpleNamespace(ranker=None))
    monkeypatch.setattr(ranker, "make_store", make_store)
    state = HubState("config.yaml")
    state._calibration_attempt_at = time.time()  # no background model fit in this test

    code, payload = state.auto_review_next("random")
    assert code == 200 and payload["item"]["profile_id"] == PID
    assert (
        payload["item"]["photo_count"] == 2
        and payload["item"]["profile_name"] == "Avery"
    )
    assert state.auto_review_image(PID, 1) == (200, PNG + b"-2", "image/png")
    assert state.auto_review_image(PID, 2)[0] == 404
    code, payload = state.auto_review_next("near_cut")
    assert payload["item"]["profile_id"] == PID2

    ok, result, msg, code = state.submit_auto_review_answer(
        {"profile_id": PID, "would_like": True}
    )
    assert (ok, code) == (True, 200)
    assert result == {"kind": "random", "label_source": "auto_review_random"}

    # An answered pass is gone; the empty random pool falls back to the near-cut one.
    code, payload = state.auto_review_next("random")
    assert payload["item"]["profile_id"] == PID2 and payload["fell_back"] is True
    # A second answer to the same pass, and an answer for a pass that never existed, are
    # conflicts that retire the card rather than becoming a second label.
    for pid in (PID, "ab" * 16):
        ok, result, _, code = state.submit_auto_review_answer(
            {"profile_id": pid, "would_like": False}
        )
        assert (ok, code, result) == (False, 409, {"retired": True})

    code, payload = state.auto_review_stats_snapshot()
    assert code == 200
    assert (
        payload["store"]["random_reviewed"] == 1
        and payload["store"]["random_would_like"] == 1
    )
    assert payload["store"]["pending_near_cut"] == 1
    assert payload["summary"]["verdict"] == "insufficient"

    # The answer is durable in a store the hub did not use, and is a training label ...
    check = SQLiteStore(db)
    try:
        rows = check.load_label_rows()
        assert (True, [0.1, 0.2, 0.3], "auto_review_random") in rows
        assert len(check.load_labels()) == 2
        # ... that the model-quality card does NOT judge the ranker on.
        assert HubState._eval_label_pairs(check) == [(True, [0.9, 0.8, 0.7])]
    finally:
        check.close()
    # Every store the hub opened for these requests was closed again (nothing left open).
    # The stats request starts the hub's background calibration, which opens a store of its own
    # and closes it when it finishes; wait for it, or this assertion races that thread (it
    # failed on about half of the runs before this wait).
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        with state._lock:
            if not state._calibration_refreshing:
                break
        time.sleep(0.01)
    assert not state._calibration_refreshing
    assert opened
    for store in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            store.con.execute("SELECT 1")


# --- HTTP ----------------------------------------------------------------------------------------
def _csrf(base):
    with urllib.request.urlopen(base + "/", timeout=5) as r:
        page = r.read().decode()
    return re.search(
        r'<meta name="operation-love-csrf" content="([^"]+)">', page
    ).group(1)


def _call(method, url, body=None, *, token=None, headers=None):
    data = None
    request_headers = dict(headers or {})
    if body is not None:
        payload = dict(body)
        if token is not None:
            payload["_csrf_token"] = token
        data = json.dumps(payload).encode()
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(
        url, data=data, method=method, headers=request_headers
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


@pytest.fixture
def served(env):
    _Handler.state = env.hub
    httpd = _bind("127.0.0.1", 8799)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(
            base=f"http://127.0.0.1:{httpd.server_address[1]}", env=env
        )
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_next_serves_json_and_refuses_bad_kinds(served):
    served.env.use(lambda: ReviewStore(queue={"random": _pass()}))
    base = served.base

    status, headers, body = _call("GET", base + "/api/auto-review/next?kind=random")
    assert status == 200 and headers["Content-Type"] == "application/json"
    assert json.loads(body)["item"]["profile_id"] == PID

    for query in [
        "",
        "?kind=",
        "?kind=all",
        "?kind=random&kind=near_cut",
        "?KIND=random",
    ]:
        status, _, body = _call("GET", base + "/api/auto-review/next" + query)
        assert status == 400, query
        assert "kind" in json.loads(body)["error"]


def test_http_image_serves_png_uncached_and_refuses_bad_parameters(served):
    served.env.use(lambda: ReviewStore(photos={(PID, 2): PNG}))
    base = served.base

    status, headers, body = _call(
        "GET", base + f"/api/auto-review/image?profile_id={PID}&index=2"
    )
    assert status == 200 and body == PNG
    assert headers["Content-Type"] == "image/png"
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Content-Length"] == str(len(PNG))

    status, _, body = _call(
        "GET", base + f"/api/auto-review/image?profile_id={PID}&index=0"
    )
    assert status == 404 and json.loads(body)["error"] == "auto pass image not found"

    bad = [
        "profile_id=../../../etc/passwd&index=0",
        f"profile_id={PID}/..&index=0",
        f"profile_id={PID.upper()}&index=0",
        f"profile_id={PID}&index=-1",
        f"profile_id={PID}&index=abc",
        f"profile_id={PID}&index=1000",
        f"profile_id={PID}",
        "index=0",
        f"profile_id={PID}&profile_id={PID2}&index=0",
        f"profile_id={PID}&index=1&index=2",
    ]
    for query in bad:
        status, _, body = _call("GET", base + "/api/auto-review/image?" + query)
        assert status == 400, query
        assert "error" in json.loads(body)
    # None of the refused requests reached the store.
    assert [c for s, _ in served.env.made for c in s.calls if c[0] == "photo"] == [
        ("photo", PID, 2),
        ("photo", PID, 0),
    ]

    status, _, _ = _call("GET", base + "/api/auto-review/nope")
    assert status == 404


def test_http_stats(served):
    served.env.use(lambda: ReviewStore(stats=_stats(random_reviewed=3)))
    served.env.hub._calibration_attempt_at = (
        time.time()
    )  # no background computation here

    status, _, body = _call("GET", served.base + "/api/auto-review/stats")

    payload = json.loads(body)
    assert status == 200 and payload["available"] is True
    assert payload["summary"]["verdict"] == "insufficient"


def test_http_answer_requires_the_csrf_token_and_a_real_boolean(served):
    store_log = []

    def factory():
        store = ReviewStore()
        store_log.append(store)
        return store

    served.env.use(factory)
    base = served.base
    url = base + "/api/auto-review/answer"
    token = _csrf(base)

    # No / wrong token: refused before anything is parsed further.
    for sent in (None, "wrong"):
        status, _, body = _call(
            "POST", url, {"profile_id": PID, "would_like": True}, token=sent
        )
        assert status == 403 and json.loads(body)["msg"] == "invalid CSRF token"
    assert store_log == []

    # Authorised but malformed: 400, still no store.
    for payload in (
        {"profile_id": PID, "would_like": "true"},
        {"profile_id": PID, "would_like": 1},
        {"profile_id": PID},
        {"profile_id": "../x", "would_like": True},
        {"profile_id": PID, "would_like": True, "extra": 1},  # unknown field
    ):
        status, _, body = _call("POST", url, payload, token=token)
        assert status == 400, payload
        assert json.loads(body)["ok"] is False
    assert store_log == []

    # A duplicated JSON key is ambiguous and refused at the parser.
    raw = f'{{"profile_id": "{PID}", "would_like": true, "would_like": false, "_csrf_token": "{token}"}}'
    request = urllib.request.Request(
        url,
        data=raw.encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(request, timeout=5)
    assert err.value.code == 400

    status, _, body = _call(
        "POST", url, {"profile_id": PID, "would_like": True}, token=token
    )
    payload = json.loads(body)
    assert status == 200 and payload["ok"] is True
    assert payload["result"] == {"kind": "random", "label_source": "auto_review_random"}
    assert [c[0] for c in store_log[0].calls] == ["answer", "flush", "close"]


def test_http_undo_requires_the_csrf_token_and_takes_no_fields(served):
    stores = []

    def factory():
        store = ReviewStore(undo=dict(_UNDONE))
        stores.append(store)
        return store

    served.env.use(factory)
    url = served.base + "/api/auto-review/undo"
    token = _csrf(served.base)

    status, _, body = _call("POST", url, {}, token=None)
    assert status == 403 and json.loads(body)["msg"] == "invalid CSRF token"
    status, _, _ = _call("POST", url, {"profile_id": PID}, token=token)
    assert (
        status == 400
    )  # undo undoes the NEWEST answer; it cannot be pointed at another
    assert stores == []

    status, _, body = _call("POST", url, {}, token=token)
    payload = json.loads(body)
    assert status == 200 and payload["ok"] is True and payload["result"] == _UNDONE
    assert [c[0] for c in stores[0].calls] == ["undo", "flush", "close"]


def test_http_answer_conflict_is_a_409_that_retires_the_card(served):
    def refuse(profile_id, would_like):
        raise ValueError("auto pass already reviewed")

    served.env.use(lambda: ReviewStore(answer=refuse))
    token = _csrf(served.base)

    status, _, body = _call(
        "POST",
        served.base + "/api/auto-review/answer",
        {"profile_id": PID, "would_like": False},
        token=token,
    )

    payload = json.loads(body)
    assert status == 409 and payload["ok"] is False
    assert (
        payload["result"] == {"retired": True} and "already reviewed" in payload["msg"]
    )


def test_http_review_routes_survive_a_run_that_already_imported_parse_qs_locally(
    served,
):
    """do_GET imports parse_qs locally in other branches; the review routes must not depend on
    that name (it is function-local there, so using it unimported is UnboundLocalError)."""
    served.env.use(lambda: ReviewStore(queue={"random": _pass()}))

    # Hit a branch that does the local import first, then a review route on the same server.
    _Handler.state = SimpleNamespace(
        training_action_snapshot=lambda **k: {"ok": True},
        auto_review_next=served.env.hub.auto_review_next,
    )
    assert _call("GET", served.base + "/api/training/checkpoint")[0] == 200
    status, _, body = _call("GET", served.base + "/api/auto-review/next?kind=random")
    assert status == 200 and json.loads(body)["available"] is True


# --- the page ------------------------------------------------------------------------------------
def _extract(name):
    m = re.search(rf"(?:async\s+)?function\s+{re.escape(name)}\s*\([^)]*\)\s*{{", _PAGE)
    assert m, f"function {name} not found in hub.html"
    start = m.end() - 1
    depth = 0
    for i in range(start, len(_PAGE)):
        if _PAGE[i] == "{":
            depth += 1
        elif _PAGE[i] == "}":
            depth -= 1
            if depth == 0:
                return _PAGE[m.start() : i + 1]
    raise AssertionError(f"unbalanced braces extracting {name}")


def _node(script):
    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")
    r = subprocess.run(
        [NODE_BIN, "-e", script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert r.returncode == 0, f"node failed:\nSTDOUT: {r.stdout}\nSTDERR: {r.stderr}"
    return json.loads(r.stdout)


_PURE = [
    "escHtml",
    "nextAutoReviewKind",
    "autoReviewPercent",
    "autoReviewScore",
    "autoReviewImageUrl",
    "autoReviewKindLabel",
    "autoReviewSourceLabel",
    "autoReviewSummaryHtml",
    "autoReviewItemHtml",
]


def _pure_js():
    return "\n".join(_extract(name) for name in _PURE) + "\n"


def test_the_card_alternates_the_two_pools():
    out = _node(
        _pure_js()
        + "console.log(JSON.stringify([nextAutoReviewKind('random'), nextAutoReviewKind('near_cut'),"
        " nextAutoReviewKind(undefined)]));"
    )
    assert out == ["near_cut", "random", "random"]


def test_the_item_html_has_one_image_per_photo_and_escapes_hostile_text():
    item = {
        "profile_id": PID,
        "kind": "near_cut",
        "app": "hinge",
        "created_at": 1_700_000_000,
        "score": 0.41,
        "threshold": 0.45,
        "photo_count": 3,
        "profile_name": '<img src=x onerror="alert(1)">',
    }
    out = _node(
        _pure_js()
        + f"console.log(JSON.stringify(autoReviewItemHtml({json.dumps(item)})));"
    )
    assert out.count("<img src=") == 3  # the hostile name is text, not a 4th image
    for i in range(3):
        assert f"/api/auto-review/image?profile_id={PID}&amp;index={i}" in out or (
            f"/api/auto-review/image?profile_id={PID}&index={i}" in out
        )
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in out
    assert "NEAR THE CUT" in out and "0.4100" in out and "0.4500" in out
    assert "(0.0400 below)" in out


def test_the_item_html_without_photos_or_an_item():
    out = _node(
        _pure_js() + "console.log(JSON.stringify([autoReviewItemHtml(null),"
        " autoReviewItemHtml({profile_id:'"
        + PID
        + "',kind:'random',photo_count:0,profile_name:''})]));"
    )
    assert "Nothing is waiting for review" in out[0]
    assert "no stored screenshots" in out[1] and "<img" not in out[1]
    assert "Unnamed profile" in out[1] and "RANDOM SAMPLE" in out[1]


def test_the_image_url_encodes_its_parameters():
    out = _node(
        _pure_js() + "console.log(JSON.stringify(autoReviewImageUrl('a b&c', 3)));"
    )
    assert out == "/api/auto-review/image?profile_id=a%20b%26c&index=3"


def _summary(stats):
    return _node(
        _pure_js()
        + f"console.log(JSON.stringify(autoReviewSummaryHtml({json.dumps(stats)})));"
    )


def _stats_payload(verdict, **over):
    payload = {
        "available": True,
        "store": {"pending_random": 4, "pending_near_cut": 7},
        "summary": {
            "random_reviewed": 40,
            "random_would_like": 10,
            "measured_false_pass_rate": 0.25,
            "observed_false_pass_lower": 0.15,
            "expected_false_pass_rate": 0.05,
            "min_random_reviews": 10,
            "verdict": verdict,
        },
        "calibration": {
            "recall_target": 0.95,
            "effective_target": 0.96,
            "threshold": 0.4712,
        },
        "calibration_source": "live_run",
        "calibration_pending": False,
        "calibration_error": None,
    }
    payload.update(over)
    return payload


def test_the_summary_shows_measured_against_expected_and_each_verdict():
    above = _summary(_stats_payload("above"))
    assert "25.0%" in above and "10 of 40" in above and "predicted 5.0%" in above
    assert "4 random sample" in above and "7 near the cut" in above
    assert "raised" in above and "15.0%" in above
    assert "Recall target 95.0%" in above and "effective 96.0%" in above
    assert "0.4712" in above and "the running bot" in above

    assert "In line with what the model predicted" in _summary(_stats_payload("within"))
    insufficient = _summary(_stats_payload("insufficient"))
    assert "Needs 10 random answers" in insufficient and "have 40" in insufficient
    assert "not available yet" in _summary(_stats_payload("unknown"))


def test_the_summary_states_pending_error_and_unavailable_calibration():
    pending = _summary(
        _stats_payload("within", calibration=None, calibration_pending=True)
    )
    assert "Calculating the predicted rate" in pending
    errored = _summary(
        _stats_payload("within", calibration=None, calibration_error="<b>no target</b>")
    )
    assert "Predicted rate unavailable: &lt;b&gt;no target&lt;/b&gt;" in errored
    unavailable = _summary({"available": False, "reason": "LegacyStore has no <x>"})
    assert "LegacyStore has no &lt;x&gt;" in unavailable
    assert "unavailable" in _summary(None).lower()


def test_every_element_the_cards_script_looks_up_exists_in_the_markup():
    script = _PAGE[
        _PAGE.index("// --- AUTO pass review card") : _PAGE.index(
            "// --- platform picker"
        )
    ]
    wanted = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", script)) | set(
        re.findall(r"'#([A-Za-z0-9_-]+)'", script)
    )
    assert {
        "autoreviewhint",
        "autoreviewbody",
        "autoreviewsummary",
        "autoreviewlike",
        "autoreviewpass",
    } <= wanted
    for element_id in wanted:
        assert f'id="{element_id}"' in _PAGE, element_id


def test_the_card_is_informational_green_and_offers_no_skip():
    card = _PAGE[_PAGE.index('id="autoreviewcard"') : _PAGE.index("live log")]
    assert "🟢" in card and "🔴" not in card
    assert ">Would have liked<" in card and ">Correct pass<" in card
    assert (
        "skip"
        not in re.sub(
            r"skipping the ones you are unsure about", "", card, flags=re.I
        ).lower()
    )
    # Run state is never touched by the card's script: it neither starts nor stops anything.
    script = _PAGE[
        _PAGE.index("// --- AUTO pass review card") : _PAGE.index(
            "// --- platform picker"
        )
    ]
    for forbidden in (
        "/api/start",
        "/api/stop",
        "renderGlobal",
        "runbanner",
        "runpill",
    ):
        assert forbidden not in script
    assert 'class="stop"' not in script and 'class="danger"' not in card


_FLOW_HARNESS = r"""
const els = {};
function mk(id){ return els[id] = els[id] || {innerHTML:'', textContent:'', disabled:false,
  querySelectorAll(){ return []; }, addEventListener(){} }; }
const $ = sel => mk(sel);
let _autoReviewItem = null, _autoReviewKind = 'random', _autoReviewBusy = false,
    _autoReviewRequest = 0, _autoReviewStatsRequest = 0, _autoReviewStatsTimer = null;
const log = [];
let nextResponses = [], postResponse = {ok:true, result:{kind:'random'}}, postGate = null;
async function getJSON(url){ log.push(['GET', url]); return nextResponses.shift(); }
async function postJSON(url, body){
  log.push(['POST', url, body]);
  if (postGate) await postGate;
  return postResponse;
}
function refreshAutoReviewStats(){ log.push(['STATS']); }
"""


def _flow(scenario):
    names = [
        "escHtml",
        "nextAutoReviewKind",
        "autoReviewPercent",
        "autoReviewScore",
        "autoReviewImageUrl",
        "autoReviewKindLabel",
        "autoReviewSourceLabel",
        "autoReviewSummaryHtml",
        "autoReviewItemHtml",
        "setAutoReviewButtons",
        "loadAutoReview",
        "answerAutoReview",
        "undoAutoReview",
    ]
    fns = "\n".join(_extract(n) for n in names)
    return _node(_FLOW_HARNESS + fns + "\n(async () => {\n" + scenario + "\n})();")


_ITEM_RANDOM = {
    "profile_id": PID,
    "kind": "random",
    "photo_count": 1,
    "profile_name": "A",
    "score": 0.4,
    "threshold": 0.5,
}
_ITEM_NEAR = dict(_ITEM_RANDOM, profile_id=PID2, kind="near_cut")


def test_answering_posts_the_boolean_then_loads_the_other_pool_and_refreshes_stats():
    out = _flow(
        f"""
        _autoReviewItem = {json.dumps(_ITEM_RANDOM)};
        nextResponses = [{{available:true, item:{json.dumps(_ITEM_NEAR)}, fell_back:false}}];
        await answerAutoReview(true);
        console.log(JSON.stringify({{log, item:_autoReviewItem, kind:_autoReviewKind, busy:_autoReviewBusy,
          hint:els['#autoreviewhint'].textContent}}));
        """
    )
    assert out["log"] == [
        ["POST", "/api/auto-review/answer", {"profile_id": PID, "would_like": True}],
        ["GET", "/api/auto-review/next?kind=near_cut"],
        ["STATS"],
    ]
    assert out["item"]["profile_id"] == PID2 and out["kind"] == "near_cut"
    assert out["busy"] is False
    assert out["hint"] == "saved as liked"


def test_a_correct_pass_posts_false_and_alternates_back_to_random():
    out = _flow(
        f"""
        _autoReviewItem = {json.dumps(_ITEM_NEAR)};
        nextResponses = [{{available:true, item:null}}];
        await answerAutoReview(false);
        console.log(JSON.stringify({{log, hint:els['#autoreviewhint'].textContent}}));
        """
    )
    assert out["log"][0] == [
        "POST",
        "/api/auto-review/answer",
        {"profile_id": PID2, "would_like": False},
    ]
    assert out["log"][1] == ["GET", "/api/auto-review/next?kind=random"]
    # The empty next pool is the more important thing to say; it is not overwritten.
    assert out["hint"] == "nothing to review"


def test_a_failed_save_keeps_the_card_and_reenables_the_buttons():
    out = _flow(
        f"""
        _autoReviewItem = {json.dumps(_ITEM_RANDOM)};
        postResponse = {{ok:false, result:null, msg:'BigQuery 503'}};
        await answerAutoReview(true);
        console.log(JSON.stringify({{log, hint:els['#autoreviewhint'].textContent,
          item:_autoReviewItem && _autoReviewItem.profile_id, busy:_autoReviewBusy,
          like:els['#autoreviewlike'].disabled, pass:els['#autoreviewpass'].disabled}}));
        """
    )
    assert [entry[0] for entry in out["log"]] == ["POST"]  # no reload, no stats
    assert out["hint"] == "BigQuery 503" and out["item"] == PID
    assert out["busy"] is False and out["like"] is False and out["pass"] is False


def test_a_retired_pass_moves_on_instead_of_sticking():
    out = _flow(
        f"""
        _autoReviewItem = {json.dumps(_ITEM_RANDOM)};
        postResponse = {{ok:false, result:{{retired:true}}, msg:'already reviewed'}};
        nextResponses = [{{available:true, item:{json.dumps(_ITEM_NEAR)}}}];
        await answerAutoReview(true);
        console.log(JSON.stringify({{log, item:_autoReviewItem.profile_id}}));
        """
    )
    assert [e[0] for e in out["log"]] == ["POST", "GET", "STATS"]
    assert out["item"] == PID2


def test_a_refused_answer_is_never_reported_as_saved():
    """A 409 'already reviewed' (another tab, or BigQuery lag re-serving an answered card) saved
    NOTHING.  Loading the next card clears the hint, and the fallback then wrote 'saved as
    liked' over the refusal -- a false confirmation of a label that does not exist."""
    out = _flow(
        f"""
        _autoReviewItem = {json.dumps(_ITEM_RANDOM)};
        postResponse = {{ok:false, result:{{retired:true}}, msg:'auto pass already reviewed'}};
        nextResponses = [{{available:true, item:{json.dumps(_ITEM_NEAR)}}}];
        await answerAutoReview(true);
        const liked = els['#autoreviewhint'].textContent;
        _autoReviewItem = {json.dumps(_ITEM_RANDOM)};
        nextResponses = [{{available:true, item:{json.dumps(_ITEM_NEAR)}}}];
        await answerAutoReview(false);
        const passed = els['#autoreviewhint'].textContent;
        _autoReviewItem = {json.dumps(_ITEM_RANDOM)};
        postResponse = {{ok:false, result:{{retired:true}}}};
        nextResponses = [{{available:true, item:{json.dumps(_ITEM_NEAR)}}}];
        await answerAutoReview(true);
        console.log(JSON.stringify({{liked, passed, bare:els['#autoreviewhint'].textContent}}));
        """
    )
    assert out["liked"] == out["passed"] == "auto pass already reviewed"
    assert out["bare"] == "could not save the answer"  # still never 'saved as ...'
    assert "saved" not in out["liked"] + out["passed"] + out["bare"].replace(
        "could not save", ""
    )


def test_a_recorded_answer_still_says_what_was_saved():
    out = _flow(
        f"""
        _autoReviewItem = {json.dumps(_ITEM_RANDOM)};
        postResponse = {{ok:true, result:{{kind:'random'}}}};
        nextResponses = [{{available:true, item:{json.dumps(_ITEM_NEAR)}}}];
        await answerAutoReview(false);
        console.log(JSON.stringify({{hint:els['#autoreviewhint'].textContent}}));
        """
    )
    assert out["hint"] == "saved as a pass"


def test_undo_posts_nothing_but_the_request_then_reloads_the_pool_the_person_came_from():
    out = _flow(
        f"""
        postResponse = {{ok:true, result:{{profile_id:'{PID}', profile_name:'Avery', kind:'near_cut', would_like:true}}}};
        nextResponses = [{{available:true, item:{json.dumps(_ITEM_NEAR)}}}];
        await undoAutoReview();
        console.log(JSON.stringify({{log, busy:_autoReviewBusy, hint:els['#autoreviewhint'].textContent,
          undoDisabled:els['#autoreviewundo'].disabled, item:_autoReviewItem && _autoReviewItem.profile_id}}));
        """
    )
    assert out["log"] == [
        ["POST", "/api/auto-review/undo", {}],
        ["GET", "/api/auto-review/next?kind=near_cut"],
        ["STATS"],
    ]
    assert out["hint"] == "undid your answer for Avery; they are back in the queue"
    assert out["busy"] is False and out["undoDisabled"] is False
    assert out["item"] == PID2


def test_undo_failures_are_shown_and_leave_the_card_alone():
    out = _flow(
        f"""
        _autoReviewItem = {json.dumps(_ITEM_RANDOM)};
        postResponse = {{ok:false, result:null, msg:'there is no review answer to undo'}};
        await undoAutoReview();
        const refused = {{hint:els['#autoreviewhint'].textContent, log:log.slice(), item:_autoReviewItem.profile_id}};
        log.length = 0;
        postGate = Promise.reject(new Error('down'));
        postGate.catch(() => {{}});
        await undoAutoReview();
        console.log(JSON.stringify({{refused, offline:els['#autoreviewhint'].textContent,
          busy:_autoReviewBusy, like:els['#autoreviewlike'].disabled}}));
        """
    )
    assert out["refused"]["hint"] == "there is no review answer to undo"
    assert [e[0] for e in out["refused"]["log"]] == ["POST"]  # no reload, no stats
    assert out["refused"]["item"] == PID  # the card the owner was on is untouched
    assert out["offline"] == "could not contact the local hub"
    assert out["busy"] is False and out["like"] is False


def test_undo_does_nothing_while_an_answer_is_in_flight():
    out = _flow(
        f"""
        _autoReviewItem = {json.dumps(_ITEM_RANDOM)};
        let release; postGate = new Promise(r => release = r);
        nextResponses = [{{available:true, item:null}}];
        const answering = answerAutoReview(true);
        await undoAutoReview();                       // while the answer is still posting
        release(); await answering;
        console.log(JSON.stringify({{posts:log.filter(e => e[0] === 'POST').map(e => e[1])}}));
        """
    )
    assert out["posts"] == ["/api/auto-review/answer"]


def test_a_double_click_posts_once_and_no_item_posts_nothing():
    out = _flow(
        f"""
        _autoReviewItem = {json.dumps(_ITEM_RANDOM)};
        let release; postGate = new Promise(r => release = r);
        nextResponses = [{{available:true, item:null}}];
        const first = answerAutoReview(true);
        const second = answerAutoReview(false);   // while the first is in flight
        release();
        await first; await second;
        const posts = log.filter(e => e[0] === 'POST').length;
        postGate = null; log.length = 0; _autoReviewItem = null;
        await answerAutoReview(true);
        console.log(JSON.stringify({{posts, afterNoItem: log.length}}));
        """
    )
    assert out == {"posts": 1, "afterNoItem": 0}


def test_loading_reports_unavailable_and_network_failures_without_throwing():
    out = _flow(
        """
        nextResponses = [{available:false, reason:'LegacyStore does not support auto pass review'}];
        await loadAutoReview('random');
        const unavailable = {body: els['#autoreviewbody'].innerHTML, hint: els['#autoreviewhint'].textContent,
                             like: els['#autoreviewlike'].disabled};
        getJSON = async () => { throw new Error('offline'); };
        await loadAutoReview('near_cut');
        console.log(JSON.stringify({unavailable, offline: els['#autoreviewbody'].innerHTML,
                                    hint: els['#autoreviewhint'].textContent}));
        """
    )
    assert "LegacyStore does not support auto pass review" in out["unavailable"]["body"]
    assert (
        out["unavailable"]["hint"] == "unavailable"
        and out["unavailable"]["like"] is True
    )
    assert "Could not contact the local hub" in out["offline"]
    assert out["hint"] == "unavailable"


def test_a_slow_older_load_cannot_overwrite_a_newer_one():
    out = _flow(
        f"""
        let releaseOld;
        const old = new Promise(r => releaseOld = r);
        let calls = 0;
        getJSON = async (url) => {{
          calls += 1;
          if (calls === 1) {{ await old; return {{available:true, item:{json.dumps(_ITEM_RANDOM)}}}; }}
          return {{available:true, item:{json.dumps(_ITEM_NEAR)}}};
        }};
        const first = loadAutoReview('random');
        await loadAutoReview('near_cut');
        releaseOld(); await first;
        console.log(JSON.stringify({{item:_autoReviewItem.profile_id}}));
        """
    )
    assert out == {"item": PID2}
