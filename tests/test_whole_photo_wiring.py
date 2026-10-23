"""Whole-photo feature wiring: decider, Training persist, AUTO archive, startup, stats, hub.

The properties pinned here:

* ``face_crop_v1`` (the default) decides exactly as before and never touches the whole-photo
  path inside ``decide()``;
* ``face_crop_whole_photo_v1`` scores ``v1 ++ whole_photo`` and a failure there, frames with
  no photo card included, stops the decision (there is no zero block);
* Training writes the label's ``photo_features`` row: in whole-photo mode as a required write
  between the label and the flush, in v1 mode best-effort (never buffered) only after the Hub
  action completed;
* an archived AUTO pass gets its row too (best-effort), while ``auto_passes`` keeps the v1
  embedding;
* every loader site (retrain, startup, ``stats``, the hub) uses the configured feature set and
  turns missing rows into the ranker's NOT READY reason, and an AUTO run that defers on it
  stops as ``feature_set_incomplete`` (hub: "whole-photo features missing"), never as a cold
  start.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import operation_love.hub as hub
import operation_love.ranker as ranker
from operation_love import stats
from operation_love import supervisor as sup
from operation_love.hub import HubState
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import (
    Decision,
    RankerDecider,
    whole_photo_block,
)
from operation_love.ranker.model import PreferenceModel
from operation_love.ranker.store import SQLiteStore
from operation_love.ranker.threshold import (
    FEATURE_SET_FACE_CROP_V1,
    FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1,
    FeatureSetIncomplete,
)
from operation_love.vision.photo_cards import (
    WHOLE_PHOTO_DIMS,
    WHOLE_PHOTO_FEATURE_VERSION,
)
from tests.test_auto_review_worker import _Decider as _AutoDecider
from tests.test_auto_review_worker import _dislike, _profile
from tests.test_auto_review_worker import _run as _run_auto
from tests.test_auto_review_worker import _Store as _AutoStore
from tests.test_recall_threshold import _run_supervisor, _SupStore
from tests.test_stats import _write_config
from tests.test_training_worker import (
    _TIMEOUT_S,
    _checkpoint,
    _new_worker,
    _submit,
    _wait_until,
)

WHOLE = FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1
SHA = "ab" * 32


@pytest.fixture(autouse=True)
def _stub_native_training_notification(monkeypatch):
    """Worker tests must never display real host notifications (or play the alert sound)."""
    import operation_love.worker as worker_module

    monkeypatch.setattr(worker_module, "notify_training_decision_ready", lambda: True)


def test_training_worker_tests_here_never_reach_the_real_host_notifier():
    """RT-5: the Training runs below would otherwise post a macOS notification and play the
    alert sound on every local run. Checked by identity, never by calling the real one."""
    import operation_love.notifications as notifications
    import operation_love.worker as worker_module

    assert (
        worker_module.notify_training_decision_ready
        is not notifications.notify_training_decision_ready
    )


REASON_TAIL = (
    "labels lack whole-photo features; run python -m tools.backfill_photo_features"
)


def _photo_vec(value=0.25):
    return [value] * WHOLE_PHOTO_DIMS


def _whole(vector=None, card_count=None, sha=SHA):
    if card_count is None:
        card_count = 0 if vector is None else 3
    return SimpleNamespace(
        vector=vector,
        card_count=card_count,
        frames_sha256=sha,
        feature_version=WHOLE_PHOTO_FEATURE_VERSION,
    )


# --- decider ---------------------------------------------------------------------------------
class _Quality:
    def __init__(self, keep=None):
        self.keep = keep

    def filter(self, photos):
        return photos[: self.keep] if self.keep is not None else photos


class _Embedder:
    def __init__(self, vec, whole=None, *, error=None):
        self.vec = vec
        self.whole = whole
        self.error = error
        self.profile_photos = None
        self.whole_calls = []

    def embed_profile(self, profile):
        self.profile_photos = list(profile.photos)
        return self.vec

    def embed_whole_photos(self, frames):
        self.whole_calls.append(list(frames))
        if self.error is not None:
            raise self.error
        return self.whole


class _ForbiddenWholeEmbedder(_Embedder):
    def embed_whole_photos(self, frames):
        raise AssertionError("v1 decide() must not compute the whole-photo embedding")


class _Model:
    def __init__(self, ready=True, decision=("like", 0.8)):
        self.ready = ready
        self.decision = decision
        self.seen = []

    def decide(self, vec):
        self.seen.append(list(vec))
        return self.decision


def _frames():
    return Profile(photos=[b"frame-1", b"frame-2", b"frame-3"], name="Avery")


@pytest.mark.parametrize(
    "decision", [("like", 0.91), ("dislike", 0.12)], ids=["like", "dislike"]
)
def test_v1_decide_is_unchanged_and_never_computes_the_whole_photo_vector(decision):
    model = _Model(decision=decision)
    embedder = _ForbiddenWholeEmbedder([1.0, 2.0, 3.0])

    d = RankerDecider(_Quality(), embedder, model).decide(_frames())

    assert d == Decision(decision[0], decision[1], [1.0, 2.0, 3.0], "ranker")
    assert (d.photo_embedding, d.photo_card_count, d.frames_sha256) == (None, 0, "")
    assert model.seen == [[1.0, 2.0, 3.0]]


def test_whole_photo_decide_scores_the_concatenation_and_keeps_v1_as_the_embedding():
    photo = _photo_vec(0.5)
    model = _Model(decision=("dislike", 0.2))
    embedder = _Embedder([1.0, 2.0], _whole(photo, card_count=4))

    d = RankerDecider(_Quality(), embedder, model, feature_set=WHOLE).decide(_frames())

    assert model.seen == [[1.0, 2.0, *photo]]
    assert d.decision == "dislike" and d.score == 0.2
    assert d.embedding == [1.0, 2.0]  # labels/auto_passes always store the v1 vector
    assert d.photo_embedding == photo and d.photo_card_count == 4
    assert d.frames_sha256 == SHA


def test_whole_photo_decide_uses_every_frame_not_the_quality_filtered_ones():
    profile = _frames()
    embedder = _Embedder([1.0], _whole(_photo_vec()))

    RankerDecider(_Quality(keep=1), embedder, _Model(), feature_set=WHOLE).decide(
        profile
    )

    assert embedder.profile_photos == [b"frame-1"]  # v1 still quality-filtered
    assert embedder.whole_calls == [
        profile.photos
    ]  # the whole-photo vector: ALL frames


def test_frames_with_no_photo_card_stop_the_decision_instead_of_a_zero_block():
    """RT-2: the embedder's NoPhotoCards propagates, and a vector-less embedding from any
    embedder is refused, so v1 ++ zeros (the v1 model in disguise) is never scored."""
    from operation_love.vision.photo_cards import NoPhotoCards

    model = _Model()
    embedder = _Embedder(
        [1.0, 2.0], error=NoPhotoCards("3 frame(s) hold no photo card")
    )
    with pytest.raises(NoPhotoCards):
        RankerDecider(_Quality(), embedder, model, feature_set=WHOLE).decide(_frames())
    embedder = _Embedder([1.0, 2.0], _whole(None, card_count=0))
    with pytest.raises(ValueError, match="no vector"):
        RankerDecider(_Quality(), embedder, model, feature_set=WHOLE).decide(_frames())
    assert model.seen == []


def test_a_whole_photo_failure_stops_the_decision():
    embedder = _Embedder([1.0], error=RuntimeError("clip broke"))

    with pytest.raises(RuntimeError, match="clip broke"):
        RankerDecider(_Quality(), embedder, _Model(), feature_set=WHOLE).decide(
            _frames()
        )


def test_an_embedder_without_whole_photo_support_stops_a_whole_photo_decision():
    class V1Only:
        def embed_profile(self, profile):
            return [1.0]

    with pytest.raises(RuntimeError, match="embed_whole_photos"):
        RankerDecider(_Quality(), V1Only(), _Model(), feature_set=WHOLE).decide(
            _frames()
        )


def test_no_face_and_cold_start_never_compute_the_whole_photo_vector():
    embedder = _ForbiddenWholeEmbedder(None)
    d = RankerDecider(_Quality(), embedder, _Model(), feature_set=WHOLE).decide(
        _frames()
    )
    assert d.decision == "no_face"

    embedder = _ForbiddenWholeEmbedder([1.0])
    d = RankerDecider(
        _Quality(), embedder, _Model(ready=False), feature_set=WHOLE
    ).decide(_frames())
    assert d.decision == "defer" and d.embedding == [1.0]


@pytest.mark.parametrize(
    "whole",
    [
        _whole([0.1] * (WHOLE_PHOTO_DIMS - 1)),
        _whole([0.1] * (WHOLE_PHOTO_DIMS + 1)),
        _whole(None, card_count=2),
        _whole(None, card_count=0),
    ],
    ids=["short", "long", "no-vector-with-cards", "zero-cards"],
)
def test_a_malformed_whole_photo_embedding_is_refused(whole):
    with pytest.raises(ValueError):
        whole_photo_block(whole)


def test_an_unknown_feature_set_is_refused_at_construction():
    with pytest.raises(ValueError, match="feature set"):
        RankerDecider(_Quality(), _Embedder([1.0]), _Model(), feature_set="whole")


class _FeatureRowsStore:
    def __init__(self, rows):
        self.rows = rows

    def load_label_rows(self):
        return [row[:3] for row in self.rows]

    def load_label_feature_rows(self, feature_version):
        assert feature_version == WHOLE_PHOTO_FEATURE_VERSION
        return list(self.rows)


def _separable_feature_rows(n=12, *, missing=0):
    rows = []
    for i in range(n):
        liked = i % 2 == 0
        sign = 1.0 if liked else -1.0
        has_row = i >= missing
        rows.append(
            (
                liked,
                [sign, sign],
                "",
                [sign * 0.01] * WHOLE_PHOTO_DIMS if has_row else None,
                has_row,
            )
        )
    return rows


def test_retrain_trains_on_the_configured_feature_set():
    model = PreferenceModel(10, min_per_class=5)
    decider = RankerDecider(None, None, model, feature_set=WHOLE)

    assert decider.retrain(_FeatureRowsStore(_separable_feature_rows())) is True
    assert model._feature_dim == 2 + WHOLE_PHOTO_DIMS

    v1 = PreferenceModel(10, min_per_class=5)
    assert RankerDecider(None, None, v1).retrain(
        _FeatureRowsStore(_separable_feature_rows())
    )
    assert v1._feature_dim == 2


def test_retrain_with_missing_rows_is_not_ready_and_drops_the_old_classifier():
    model = PreferenceModel(10, min_per_class=5)
    decider = RankerDecider(None, None, model, feature_set=WHOLE)
    assert decider.retrain(_FeatureRowsStore(_separable_feature_rows())) is True

    ready = decider.retrain(_FeatureRowsStore(_separable_feature_rows(missing=3)))

    assert ready is False and model.ready is False
    assert model.not_ready_reason == f"3 of 12 {REASON_TAIL}"


# --- Training persist --------------------------------------------------------------------------
class _PhotoDecider:
    """test_training_worker's decider plus the whole-photo capability."""

    def __init__(self, base, *, feature_set=FEATURE_SET_FACE_CROP_V1, error=None):
        self.base = base
        self.feature_set = feature_set
        self.error = error
        self.whole_profiles = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def embed_whole_photos(self, profile):
        self.whole_profiles.append(profile)
        self.base.events.append(("whole_photo",))
        if self.error is not None:
            raise self.error
        return _whole(_photo_vec(), card_count=2)


def _with_photo_store(store, *, result=True, error=None):
    rows = []

    def add_photo_features(**row):
        store.events.append(("photo_features", row["source"]))
        rows.append(row)
        if error is not None:
            raise error
        return result

    store.add_photo_features = add_photo_features
    return rows


def _training_run(
    feature_set, *, embed_error=None, store_result=True, store_error=None
):
    worker, driver, decider, opener, store, bridge, events = _new_worker()
    photo_decider = _PhotoDecider(decider, feature_set=feature_set, error=embed_error)
    worker.decider = photo_decider
    rows = _with_photo_store(store, result=store_result, error=store_error)
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()
    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    return SimpleNamespace(
        store=store, rows=rows, events=events, result=result, decider=photo_decider
    )


def _expected_training_row(run, **extra):
    return {
        "run_id": "training-run",
        "app": "hinge",
        "profile_id": run.store.profiles[0][0],
        "feature_version": WHOLE_PHOTO_FEATURE_VERSION,
        "embedding": _photo_vec(),
        "card_count": 2,
        "frames_sha256": SHA,
        "source": "live",
        **extra,
    }


def test_whole_photo_training_writes_the_labels_photo_row_before_the_flush():
    run = _training_run(WHOLE)

    assert run.result["status"] == "completed"
    assert len(run.store.labels) == 1 and run.store.labels[0][1] == [0.25, 0.75]
    assert run.rows == [_expected_training_row(run)]  # buffered: no best_effort
    order = [e[0] for e in run.events]
    assert order.index("whole_photo") < order.index(
        "profile"
    )  # computed before any write
    assert order.index("label") < order.index("photo_features") < order.index("flush")
    assert order.index("flush") < run.events.index(("hub_complete", "completed"))


def test_v1_training_writes_the_photo_row_best_effort_after_the_hub_action_completed():
    """RT-1/RT-3: in v1 mode nothing reads the row, so it can neither fail nor delay the
    Hub action: computed and written (never buffered) only once the action is complete."""
    run = _training_run(FEATURE_SET_FACE_CROP_V1)

    assert run.result["status"] == "completed"
    assert len(run.store.labels) == 1 and run.store.labels[0][1] == [0.25, 0.75]
    assert run.rows == [_expected_training_row(run, best_effort=True)]
    order = run.events
    completed = order.index(("hub_complete", "completed"))
    assert order.index(("flush",)) < completed
    assert (
        completed
        < order.index(("whole_photo",))
        < order.index(("photo_features", "live"))
    )


def test_v1_training_with_no_photo_card_keeps_the_label_and_writes_no_row(capsys):
    from operation_love.vision.photo_cards import NoPhotoCards

    run = _training_run(
        FEATURE_SET_FACE_CROP_V1,
        embed_error=NoPhotoCards("12 frame(s) hold no photo card"),
    )

    assert run.result["status"] == "completed"
    assert len(run.store.labels) == 1 and run.rows == []
    out = capsys.readouterr().out
    assert "WARNING" in out and "no photo card" in out
    assert "python -m tools.backfill_photo_features" in out


def test_v1_training_skips_a_store_that_cannot_write_best_effort(capsys):
    worker, driver, decider, opener, store, bridge, events = _new_worker()
    photo_decider = _PhotoDecider(decider)
    worker.decider = photo_decider
    rows = []

    def add_photo_features(
        *,
        run_id,
        app,
        profile_id,
        feature_version,
        embedding,
        card_count,
        frames_sha256,
        source,
    ):
        rows.append(profile_id)
        return True

    store.add_photo_features = add_photo_features
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)

    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert result["status"] == "completed" and rows == []
    assert (
        photo_decider.whole_profiles == []
    )  # nothing computed for a row it cannot take
    assert "cannot write a best-effort row" in capsys.readouterr().out


def test_v1_training_keeps_the_label_when_the_whole_photo_vector_fails(capsys):
    run = _training_run(
        FEATURE_SET_FACE_CROP_V1, embed_error=RuntimeError("clip broke")
    )

    assert run.result["status"] == "completed"
    assert len(run.store.labels) == 1
    assert run.rows == []
    out = capsys.readouterr().out
    assert "WARNING" in out and "clip broke" in out
    assert "python -m tools.backfill_photo_features" in out


@pytest.mark.parametrize(
    "store_kwargs",
    [{"store_result": False}, {"store_error": RuntimeError("bq down")}],
    ids=["returned-false", "raised"],
)
def test_v1_training_keeps_the_label_when_the_photo_row_cannot_be_saved(
    store_kwargs, capsys
):
    run = _training_run(FEATURE_SET_FACE_CROP_V1, **store_kwargs)

    assert run.result["status"] == "completed"
    assert len(run.store.labels) == 1
    assert ("flush",) in run.events
    assert "WARNING" in capsys.readouterr().out


def test_whole_photo_training_fails_the_persist_when_the_vector_fails():
    run = _training_run(WHOLE, embed_error=RuntimeError("clip broke"))

    assert run.result["status"] == "failed"
    # Failed before any durable write: no archive, no decision, no label.
    assert run.store.profiles == [] and run.store.labels == []
    assert run.store.decisions == [] and run.rows == []


@pytest.mark.parametrize(
    "store_kwargs",
    [{"store_result": False}, {"store_error": RuntimeError("bq down")}],
    ids=["returned-false", "raised"],
)
def test_whole_photo_training_fails_the_persist_when_the_photo_row_fails(store_kwargs):
    run = _training_run(WHOLE, **store_kwargs)

    assert run.result["status"] == "failed"
    assert ("flush",) not in run.events  # never reported durable


def test_whole_photo_training_refuses_a_store_that_cannot_hold_the_row():
    worker, driver, decider, opener, store, bridge, events = _new_worker()
    worker.decider = _PhotoDecider(decider, feature_set=WHOLE)
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)

    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert result["status"] == "failed"
    assert store.profiles == [] and store.labels == []


def test_v1_training_with_a_store_lacking_the_table_computes_nothing():
    worker, driver, decider, opener, store, bridge, events = _new_worker()
    photo_decider = _PhotoDecider(decider)
    worker.decider = photo_decider
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)

    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    assert result["status"] == "completed"
    assert photo_decider.whole_profiles == []  # no latency spent on an unstorable row


# --- AUTO pass archive -------------------------------------------------------------------------
class _PhotoAutoDecider(_AutoDecider):
    def __init__(self, decisions, *, feature_set=FEATURE_SET_FACE_CROP_V1, error=None):
        super().__init__(decisions)
        self.feature_set = feature_set
        self.error = error
        self.whole_profiles = []

    def embed_whole_photos(self, profile):
        self.whole_profiles.append(profile)
        if self.error is not None:
            raise self.error
        return _whole(_photo_vec(0.75), card_count=5, sha="cd" * 32)


def test_an_archived_v1_pass_gets_its_photo_row_and_keeps_v1_in_auto_passes():
    events = []
    store = _AutoStore(events)
    rows = []
    store.add_photo_features = lambda **row: rows.append(row) or True
    profile = _profile(1)
    decider = _PhotoAutoDecider([_dislike(0.3, embedding=(0.1, 0.2))])

    _run_auto([profile], decider, store=store, events=events)

    assert store.auto_passes[0]["embedding"] == [0.1, 0.2]
    assert decider.whole_profiles == [profile]
    assert rows == [
        {
            "run_id": "run-1",
            "app": "bumble",
            "profile_id": store.auto_passes[0]["profile_id"],
            "feature_version": WHOLE_PHOTO_FEATURE_VERSION,
            "embedding": _photo_vec(0.75),
            "card_count": 5,
            "frames_sha256": "cd" * 32,
            "source": "live",
            "best_effort": True,  # never buffered: cannot fail the shutdown flush
        }
    ]
    assert store.labels == []  # an unreviewed pass is still never a label


def test_an_archived_whole_photo_pass_reuses_the_decisions_vector():
    events = []
    store = _AutoStore(events)
    rows = []
    store.add_photo_features = lambda **row: rows.append(row) or True
    decision = Decision(
        "dislike",
        0.3,
        [0.1, 0.2],
        "ranker",
        photo_embedding=_photo_vec(0.5),
        photo_card_count=4,
        frames_sha256="ef" * 32,
    )
    decider = _PhotoAutoDecider([decision], feature_set=WHOLE)

    _run_auto([_profile(1)], decider, store=store, events=events)

    assert decider.whole_profiles == []  # computed once, in decide()
    assert store.auto_passes[0]["embedding"] == [0.1, 0.2]
    assert rows[0]["embedding"] == _photo_vec(0.5) and rows[0]["card_count"] == 4
    assert rows[0]["frames_sha256"] == "ef" * 32
    assert rows[0]["profile_id"] == store.auto_passes[0]["profile_id"]
    assert rows[0]["best_effort"] is True


@pytest.mark.parametrize("where", ["embed", "store"])
def test_a_failed_pass_photo_row_never_stops_the_run(where, capsys):
    events = []
    store = _AutoStore(events)

    def broken(**row):
        raise RuntimeError("store broke")

    store.add_photo_features = (
        broken if where == "store" else (lambda **row: pytest.fail("not computed"))
    )
    decider = _PhotoAutoDecider(
        [_dislike(0.3), _dislike(0.2)],
        error=RuntimeError("clip broke") if where == "embed" else None,
    )

    _run_auto([_profile(1), _profile(2)], decider, store=store, events=events)

    assert len(store.auto_passes) == 2  # both passes archived, the run went on
    out = capsys.readouterr().out
    assert out.count("WARNING") == 2
    assert ("clip broke" if where == "embed" else "store broke") in out


def test_a_rejected_archive_writes_no_photo_row():
    events = []
    store = _AutoStore(events, auto_pass=lambda **kw: False)
    rows = []
    store.add_photo_features = lambda **row: rows.append(row) or True

    _run_auto(
        [_profile(1)], _PhotoAutoDecider([_dislike()]), store=store, events=events
    )

    assert rows == []


# --- startup, stats and hub use the configured feature set ------------------------------------
def test_supervisor_passes_the_feature_set_and_reports_missing_rows(
    monkeypatch, tmp_path, capsys
):
    built = []
    real = sup.RankerDecider

    class SpyDecider(real):
        def __init__(self, *a, **kw):
            built.append(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(sup, "RankerDecider", SpyDecider)

    class Store(_SupStore):
        def load_label_feature_rows(self, feature_version):
            return [
                (True, [1.0], "", None, False),
                (False, [2.0], "", _photo_vec(), True),
            ]

    spy, snap = _run_supervisor(
        monkeypatch,
        tmp_path,
        Store(rows=[(True, [1.0], ""), (False, [2.0], "")]),
        extra=f"  feature_set: {WHOLE}",
    )

    assert built == [{"feature_set": WHOLE}]
    assert spy.models[0]["trained"] == []  # never trained on a partial label set
    assert snap["labels"] == 2 and snap["ranker_ready"] is False
    out = capsys.readouterr().out
    assert "Ranker: labels=2 ready=False" in out
    assert f"Ranker not ready: 1 of 2 {REASON_TAIL}" in out


def test_supervisor_defaults_to_v1(monkeypatch, tmp_path, capsys):
    built = []
    real = sup.RankerDecider

    class SpyDecider(real):
        def __init__(self, *a, **kw):
            built.append(kw)
            super().__init__(*a, **kw)

    monkeypatch.setattr(sup, "RankerDecider", SpyDecider)
    rows = [(True, [1.0], ""), (False, [2.0], "")]

    spy, _ = _run_supervisor(monkeypatch, tmp_path, _SupStore(rows=rows))

    assert built == [{"feature_set": FEATURE_SET_FACE_CROP_V1}]
    assert spy.models[0]["trained"][0][0] == rows
    assert "feature set" not in capsys.readouterr().out


def test_stats_reports_missing_whole_photo_rows(tmp_path, capsys):
    path = _write_config(
        tmp_path,
        ranker={"min_labels_to_engage": 2, "min_per_class": 1, "feature_set": WHOLE},
    )
    store = SQLiteStore(tmp_path / "store.db")
    store.add_label("r1", "bumble", True, [1.0, 1.0], profile_id="p1")
    store.add_label("r1", "bumble", False, [-1.0, -1.0], profile_id="p2")
    store.add_photo_features(
        run_id="r1",
        app="bumble",
        profile_id="p1",
        feature_version=WHOLE_PHOTO_FEATURE_VERSION,
        embedding=_photo_vec(),
        card_count=3,
        frames_sha256=SHA,
        source="live",
    )
    store.close()

    stats.show(str(path))

    out = capsys.readouterr().out
    assert "Labels  : 2  (liked 1 / passed 1)" in out
    assert f"Features: {WHOLE}" in out
    assert "Features: whole-photo rows for 1 of 2 labels (1 missing, 0 with zero" in out
    assert "Ranker  : ready=False" in out
    assert f"Not ready: 1 of 2 {REASON_TAIL}" in out


def test_stats_counts_a_stored_zero_card_row_as_missing(tmp_path, capsys):
    path = _write_config(
        tmp_path,
        ranker={"min_labels_to_engage": 2, "min_per_class": 1, "feature_set": WHOLE},
    )
    store = SQLiteStore(tmp_path / "store.db")
    store.add_label("r1", "bumble", True, [1.0, 1.0], profile_id="p1")
    store.add_label("r1", "bumble", False, [-1.0, -1.0], profile_id="p2")
    store.add_photo_features(
        run_id="r1",
        app="bumble",
        profile_id="p2",
        feature_version=WHOLE_PHOTO_FEATURE_VERSION,
        embedding=_photo_vec(),
        card_count=3,
        frames_sha256=SHA,
        source="live",
    )
    store.con.execute(  # a zero-card row as an earlier build stored it
        "INSERT INTO photo_features (run_id, app, profile_id, created_at, feature_version,"
        " embedding, card_count, frames_sha256, source)"
        " VALUES ('r1','bumble','p1',1.0,?,'[]',0,?,'live')",
        (WHOLE_PHOTO_FEATURE_VERSION, SHA),
    )
    store.con.commit()
    store.close()

    stats.show(str(path))

    out = capsys.readouterr().out
    assert "Features: whole-photo rows for 2 of 2 labels (0 missing, 1 with zero" in out
    assert "Ranker  : ready=False" in out
    assert (
        "Not ready: 1 of 2 labels lack whole-photo features (1 of them stored with zero "
        "photo cards)"
    ) in out


def test_stats_trains_on_the_whole_photo_rows_when_complete(tmp_path, capsys):
    path = _write_config(
        tmp_path,
        ranker={"min_labels_to_engage": 2, "min_per_class": 1, "feature_set": WHOLE},
    )
    store = SQLiteStore(tmp_path / "store.db")
    for pid, liked in (("p1", True), ("p2", False)):
        store.add_label("r1", "bumble", liked, [1.0 if liked else -1.0], profile_id=pid)
        store.add_photo_features(
            run_id="r1",
            app="bumble",
            profile_id=pid,
            feature_version=WHOLE_PHOTO_FEATURE_VERSION,
            embedding=_photo_vec(0.1 if liked else -0.1),
            card_count=1,
            frames_sha256=SHA,
            source="backfill",
        )
    store.close()

    stats.show(str(path))

    out = capsys.readouterr().out
    assert "Ranker  : ready=True" in out
    assert "(0 missing, 0 with zero photo cards)" in out


def test_stats_v1_output_has_no_feature_lines(tmp_path, capsys):
    path = _write_config(tmp_path, ranker={"min_labels_to_engage": 40})
    stats.show(str(path))
    assert "Features" not in capsys.readouterr().out


class _HubStore:
    def __init__(self, rows):
        self.rows = rows
        self.closed = 0

    def load_label_rows(self):
        return [row[:3] for row in self.rows]

    def load_label_feature_rows(self, feature_version):
        return list(self.rows)

    def photo_feature_coverage(self, feature_version):
        with_rows = sum(1 for row in self.rows if row[4])
        return {
            "labels": len(self.rows),
            "with_features": with_rows,
            "missing": len(self.rows) - with_rows,
            "zero_cards": sum(1 for row in self.rows if row[4] and row[3] is None),
        }

    def auto_review_stats(self, *, app=None):
        return None

    def close(self):
        self.closed += 1


def test_hub_eval_pairs_use_the_feature_set_and_stay_manual_only():
    store = _HubStore(
        [
            (True, [1.0], "", _photo_vec(0.5), True),
            (False, [2.0], "auto_review_random", _photo_vec(0.25), True),
            (False, [3.0], "manual", _photo_vec(-0.5), True),
        ]
    )

    assert HubState._eval_label_pairs(store) == [(True, [1.0]), (False, [3.0])]
    assert HubState._eval_label_pairs(store, WHOLE) == [
        (True, [1.0, *_photo_vec(0.5)]),
        (False, [3.0, *_photo_vec(-0.5)]),
    ]
    # A stored zero-card row is incomplete, never a zero block.
    store.rows[2] = (False, [3.0], "manual", None, True)
    with pytest.raises(FeatureSetIncomplete) as info:
        HubState._eval_label_pairs(store, WHOLE)
    assert info.value.zero_cards == 1


def _hub_with(monkeypatch, store, feature_set):
    ranker_cfg = SimpleNamespace(
        like_threshold=0.5,
        like_recall_target=0.95,
        min_labels_to_engage=2,
        min_per_class=1,
        contextual_recall_budget=0.02,
        feature_set=feature_set,
    )
    monkeypatch.setattr(
        hub.cfg_mod, "load", lambda path: SimpleNamespace(ranker=ranker_cfg)
    )
    monkeypatch.setattr(ranker, "make_store", lambda cfg, ensure=True: store)
    return HubState("config.yaml")


def test_hub_model_quality_card_reports_missing_rows_with_coverage(monkeypatch):
    import operation_love.ranker.evaluate as eval_mod

    monkeypatch.setattr(
        eval_mod, "evaluate", lambda *a, **k: pytest.fail("must not evaluate")
    )
    store = _HubStore(
        [(True, [1.0], "", None, False), (False, [2.0], "", _photo_vec(), True)]
    )
    state = _hub_with(monkeypatch, store, WHOLE)

    snapshot = state.eval_snapshot()

    assert snapshot["status"] == "incomplete_features"
    assert f"1 of 2 {REASON_TAIL}" in snapshot["message"]
    assert "whole-photo rows for 1 of 2 labels (1 missing" in snapshot["message"]
    assert store.closed == 1


def test_hub_review_calibration_names_missing_rows(monkeypatch):
    store = _HubStore([(True, [1.0], "", None, False)])
    state = _hub_with(monkeypatch, store, WHOLE)

    state._compute_review_calibration()

    assert state._calibration_cache is None
    assert state._calibration_error == f"1 of 1 {REASON_TAIL}"


def test_feature_set_incomplete_reason_is_the_one_every_surface_prints():
    exc = FeatureSetIncomplete(missing=7, labels=500)
    assert exc.reason == f"7 of 500 {REASON_TAIL}"


def _defer_run(model, feature_set=WHOLE):
    from operation_love.status import RunStatus

    status = RunStatus("run-1", ["bumble"], min_labels=1, mode="auto")
    decider = SimpleNamespace(
        feature_set=feature_set,
        model=model,
        decide=lambda profile: Decision("defer", 0.0, [1.0], "cold_start"),
    )
    _run_auto([_profile(1)], decider, store=_AutoStore([]), status=status)
    return status.snapshot()["apps"]["bumble"]


def test_the_worker_names_the_reason_when_a_whole_photo_ranker_defers(capsys):
    model = SimpleNamespace(
        threshold=0.5,
        not_ready_reason=f"2 of 9 {REASON_TAIL}",
        not_ready_kind="feature_set_incomplete",
        ready=False,
    )

    view = _defer_run(model)

    out = capsys.readouterr().out
    assert f"Ranker not ready: 2 of 9 {REASON_TAIL}" in out
    assert "cold-start" not in out and "run training mode" not in out
    assert (view["state"], view["last_decision"]) == ("stopped", "defer")
    assert view["stop_kind"] == "feature_set_incomplete"
    assert view["stop_reason"] == f"2 of 9 {REASON_TAIL}"


@pytest.mark.parametrize("feature_set", [WHOLE, FEATURE_SET_FACE_CROP_V1])
def test_a_genuine_cold_start_still_stops_as_one(feature_set, capsys):
    """Only the structured not_ready_kind says "missing features": a whole-photo ranker that
    is simply short of labels (its reason set, its kind None) keeps today's cold-start stop."""
    model = SimpleNamespace(
        threshold=0.5,
        not_ready_reason="3 labels < min_labels 40",
        not_ready_kind=None,
        ready=False,
    )

    view = _defer_run(model, feature_set)

    out = capsys.readouterr().out
    assert "ranker not ready (cold-start)" in out and "Ranker not ready:" not in out
    assert (view["state"], view["last_decision"]) == ("stopped", "defer")
    assert view["stop_kind"] is None and view["stop_reason"] is None


def test_a_real_auto_worker_on_a_store_missing_a_row_stops_naming_the_backfill(
    tmp_path, capsys
):
    """The seam end to end: real SQLiteStore (20 labels with rows + 1 without), real
    RankerDecider and PreferenceModel, real Worker in AUTO."""
    from operation_love.status import RunStatus

    store = SQLiteStore(tmp_path / "store.db")
    try:
        for i in range(21):
            liked = i % 2 == 0
            store.add_label(
                "r1", "bumble", liked, [1.0 if liked else -1.0] * 2, profile_id=f"p{i}"
            )
            if i < 20:
                store.add_photo_features(
                    run_id="r1",
                    app="bumble",
                    profile_id=f"p{i}",
                    feature_version=WHOLE_PHOTO_FEATURE_VERSION,
                    embedding=_photo_vec(0.1 if liked else -0.1),
                    card_count=2,
                    frames_sha256=SHA,
                    source="backfill",
                )
        model = PreferenceModel(10, min_per_class=5)
        decider = RankerDecider(
            _Quality(), _Embedder([1.0, 1.0]), model, feature_set=WHOLE
        )
        assert decider.retrain(store) is False
        status = RunStatus("run-1", ["bumble"], min_labels=1, mode="auto")
        _run_auto([_profile(1)], decider, store=_AutoStore([]), status=status)
    finally:
        store.close()

    view = status.snapshot()["apps"]["bumble"]
    reason = f"1 of 21 {REASON_TAIL}"
    assert view["stop_kind"] == "feature_set_incomplete"
    assert view["stop_reason"] == reason
    out = capsys.readouterr().out
    assert f"Ranker not ready: {reason}" in out and "cold-start" not in out


def test_run_status_accepts_the_feature_set_incomplete_stop_kind():
    from operation_love.status import RunStatus

    status = RunStatus("run-1", ["hinge"], min_labels=1, mode="auto")
    status.set_app(
        "hinge",
        state="stopped",
        last_decision="defer",
        stop_kind="feature_set_incomplete",
        stop_reason="1 of 2 labels lack whole-photo features",
    )
    assert status.snapshot()["apps"]["hinge"]["stop_kind"] == "feature_set_incomplete"
    with pytest.raises(ValueError):
        status.set_app("hinge", stop_kind="feature_set_incompletE")


# --- the hub names the remedy -----------------------------------------------------------------
def _node_or_skip():
    from tests.test_hub import NODE_BIN

    if NODE_BIN is None:
        pytest.skip("node is not available on this machine")


@pytest.mark.parametrize("with_reason", [True, False], ids=["reason", "no-reason"])
def test_hub_run_box_says_features_are_missing_not_cold_start(with_reason):
    _node_or_skip()
    from tests.test_hub import _run_node, _runstatus_script

    app = {
        "app": "hinge",
        "mode": "auto",
        "state": "stopped",
        "last_decision": "defer",
        "stop_kind": "feature_set_incomplete",
        "swipes_run": 0,
    }
    if with_reason:
        app["stop_reason"] = f"3 of 40 {REASON_TAIL}"
    result = _run_node(
        _runstatus_script({"running": True, "status": {"apps": {"hinge": app}}})
    )
    assert "whole-photo features missing" in result["html"]
    assert "cold-start" not in result["html"]
    assert "run training mode first" not in result["html"]
    if with_reason:
        assert "python -m tools.backfill_photo_features" in result["html"]


def test_hub_run_box_keeps_the_cold_start_wording_for_a_plain_defer():
    _node_or_skip()
    from tests.test_hub import _run_node, _runstatus_script

    app = {
        "app": "hinge",
        "mode": "auto",
        "state": "stopped",
        "last_decision": "defer",
        "swipes_run": 0,
    }
    result = _run_node(
        _runstatus_script({"running": True, "status": {"apps": {"hinge": app}}})
    )
    assert (
        "cold-start" in result["html"] and "run training mode first" in result["html"]
    )


def test_hub_model_quality_card_says_backfill_needed_for_incomplete_features():
    _node_or_skip()
    from tests.test_auto_review_hub import _render_eval

    message = (
        f"The ranker is not ready: 1 of 2 {REASON_TAIL}. Coverage: whole-photo rows for "
        "1 of 2 labels (1 missing, 0 with zero photo cards)."
    )
    out = _render_eval({"status": "incomplete_features", "message": message})
    assert out["badge"] == "BACKFILL NEEDED" and "not-ready" in out["cls"]
    assert "KEEP TRAINING" not in out["badge"]
    assert "Not enough evidence yet" not in out["body"]
    assert "Whole-photo features missing" in out["body"]
    assert "python -m tools.backfill_photo_features" in out["body"]
    # every other non-ok status keeps the learning card
    other = _render_eval({"status": "insufficient_data", "training": {"labels": 3}})
    assert other["badge"] == "KEEP TRAINING"
