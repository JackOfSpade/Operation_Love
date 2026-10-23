"""Whole-photo feature set, end to end across the four builders' seams.

Every other whole-photo test file exercises one module against doubles of its neighbours.  Here
the neighbours are REAL: a real ``SQLiteStore`` (and, where Cloud Storage frames are needed, the
real ``BigQueryStore`` over the SQL-executing double of tests/test_photo_features_store.py), the
real ``Worker`` Training persist and AUTO archive paths, the real ``RankerDecider`` /
``PreferenceModel``, ``threshold.fetch_label_rows``, ``stats``, the supervisor's startup, the
hub's eval pairs, and both tools' real entry points.  Only the phone, the opener, the vision
models and Cloud Storage are faked; the fake embedder returns the real ``WholePhotoEmbedding``
type with the real ``photo_cards.frames_sha256`` digest, so the digest seam between the live
row and the backfill is checked, not assumed.
"""

from __future__ import annotations

import hashlib
import json
import threading

import numpy as np
import pytest

import operation_love.vision.embed as embed_module
import operation_love.worker as worker_module
from operation_love import stats
from operation_love.hub import HubState
from operation_love.perception.capture import Profile
from operation_love.ranker.bigquery_store import BigQueryStore
from operation_love.ranker.decider import RankerDecider
from operation_love.ranker.evaluate import identity_groups
from operation_love.ranker.model import PreferenceModel
from operation_love.ranker.store import SQLiteStore
from operation_love.ranker.threshold import (
    FEATURE_SET_FACE_CROP_V1,
    FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1,
    FeatureSetIncomplete,
    fetch_label_rows,
    fetch_training_inputs,
)
from operation_love.vision.embed import WholePhotoEmbedding
from operation_love.vision.photo_cards import (
    NoPhotoCards,
    WHOLE_PHOTO_DIMS,
    WHOLE_PHOTO_FEATURE_VERSION,
    frames_sha256,
)
from operation_love.worker import Worker
from tests.test_auto_review_worker import _Driver as _AutoDriver
from tests.test_photo_features_store import _Bucket, _SqlBQ
from tests.test_photo_features_store import (
    _bigquery_sdk_stub as _bigquery_sdk_stub_fixture,
)
from tests.test_recall_threshold import _run_supervisor
from tests.test_stats import _write_config
from tests.test_training_worker import (
    _FRAME,
    _TIMEOUT_S,
    _checkpoint,
    _Opener,
    _Pacing,
    _RecordingBridge,
    _submit,
    _TrainingDriver,
    _wait_until,
)
from tools import backfill_photo_features as backfill_tool
from tools import compare_feature_sets as compare_tool

V1 = FEATURE_SET_FACE_CROP_V1
WHOLE = FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1
VERSION = WHOLE_PHOTO_FEATURE_VERSION
V1_DIMS = 1280  # 512-d ArcFace identity template + 768-d face-crop CLIP
BACKFILL = "python -m tools.backfill_photo_features"

# The BigQuery SQL double needs the SDK stand-in fixture from its own module.
_bigquery_sdk_stub = _bigquery_sdk_stub_fixture


@pytest.fixture(autouse=True)
def _no_host_notifications(monkeypatch):
    monkeypatch.setattr(worker_module, "notify_training_decision_ready", lambda: True)


# --------------------------------------------------------------------------------------------
# Deterministic stand-ins for the vision models.
# --------------------------------------------------------------------------------------------
def _seed(tag: str) -> int:
    return int(hashlib.sha256(tag.encode()).hexdigest()[:12], 16)


def _v1_vector(name: str) -> list[float]:
    """One distinct person per name (dims 0-511), the like signal in the face-crop part."""
    rng = np.random.default_rng(_seed(f"v1:{name}"))
    identity = rng.standard_normal(512)
    identity /= np.linalg.norm(identity)
    sign = 1.0 if name.startswith("L") else -1.0
    rest = rng.standard_normal(V1_DIMS - 512) * 0.02 + sign * 0.03
    return [float(x) for x in (*identity, *rest)]


def _photo_vector(frames: list[bytes]) -> list[float]:
    rng = np.random.default_rng(_seed("photo:" + frames_sha256(frames)))
    sign = 1.0 if frames[0][:1] == b"L" else -1.0
    v = rng.standard_normal(WHOLE_PHOTO_DIMS) * 0.02 + sign * 0.03
    v /= np.linalg.norm(v)
    return [float(x) for x in v]


class _Embedder:
    """``Embedder``'s two entry points; the whole-photo result is the REAL dataclass."""

    def __init__(self, *, zero_cards: bool = False):
        self.zero_cards = zero_cards
        self.whole_calls: list[list[bytes]] = []
        self.warmed = 0
        self._lock = threading.Lock()

    def warmup(self):
        self.warmed += 1

    def embed_profile(self, profile):
        return _v1_vector(profile.name)

    def embed_whole_photos(self, frames):
        frames = list(frames)
        with self._lock:
            self.whole_calls.append(frames)
        digest = frames_sha256(frames)
        if self.zero_cards:  # the live contract: no photo card is an extraction failure
            raise NoPhotoCards(f"{len(frames)} frame(s) hold no photo card")
        return WholePhotoEmbedding(
            vector=_photo_vector(frames), card_count=len(frames), frames_sha256=digest
        )


class _AllPhotos:
    def filter(self, photos):
        return list(photos)


def _frames_for(name: str, n: int = 3) -> list[bytes]:
    return [f"{name}-frame-{k}".encode() for k in range(n)]


def _seed_labels(store, n: int, *, app: str = "hinge", rows: bool = True) -> list[str]:
    """``n`` manual labels (alternating like/pass), each with its photo row unless not ``rows``."""
    embedder = _Embedder()
    ids = []
    for i in range(n):
        name = f"{'L' if i % 2 == 0 else 'P'}seed{i}"
        profile_id = f"seed-{i:03d}"
        store.add_label(
            "seed-run",
            app,
            name.startswith("L"),
            _v1_vector(name),
            source="manual",
            profile_id=profile_id,
        )
        if rows:
            whole = embedder.embed_whole_photos(_frames_for(name))
            assert (
                store.add_photo_features(
                    run_id="seed-run",
                    app=app,
                    profile_id=profile_id,
                    feature_version=whole.feature_version,
                    embedding=whole.vector,
                    card_count=whole.card_count,
                    frames_sha256=whole.frames_sha256,
                    source="backfill",
                )
                is True
            )
        ids.append(profile_id)
    return ids


def _sqlite_rows(store, sql: str, params=()):
    with store._lock:
        return store.con.execute(sql, params).fetchall()


def _photo_table(store):
    return _sqlite_rows(
        store,
        "SELECT run_id, app, profile_id, feature_version, embedding, card_count,"
        " frames_sha256, source FROM photo_features ORDER BY id",
    )


# --------------------------------------------------------------------------------------------
# Training: the real worker persist path writes the label AND its photo row.
# --------------------------------------------------------------------------------------------
def _run_training(store, decider, command="like"):
    events = []
    driver = _TrainingDriver(events)
    bridge = _RecordingBridge(events)
    worker = Worker(
        "hinge",
        driver,
        decider,
        _Opener(events),
        store,
        "training-run",
        _Pacing(),
        threading.Event(),
        mode="training",
        training_action_bridge=bridge,
    )
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, command)
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()
    return bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]


@pytest.mark.parametrize("feature_set", [V1, WHOLE])
def test_frames_with_no_photo_card_write_no_row_on_a_real_sqlite_store(
    tmp_path, feature_set
):
    """RT-2: v1 Training keeps the label with NO row (the backfill reports it later), and
    whole-photo Training fails the action before any durable write."""
    store = SQLiteStore(tmp_path / "store.db")
    embedder = _Embedder(zero_cards=True)
    decider = RankerDecider(
        _AllPhotos(), embedder, PreferenceModel(40), feature_set=feature_set
    )
    try:
        result = _run_training(store, decider, "like")
        assert embedder.whole_calls == [[_FRAME]]
        assert _photo_table(store) == []
        if feature_set == V1:
            assert result["status"] == "completed", result
            assert store.load_label_rows() == [(True, _v1_vector("Ari"), "manual")]
            assert store.photo_feature_coverage(VERSION)["missing"] == 1
            [entry] = store.labels_missing_photo_features(VERSION)
            assert entry["source"] == "manual"
            with pytest.raises(FeatureSetIncomplete, match="1 of 1"):
                fetch_label_rows(store, WHOLE)
        else:
            assert result["status"] == "failed", result
            assert store.load_label_rows() == []
    finally:
        store.close()


@pytest.mark.parametrize("feature_set", [V1, WHOLE])
def test_training_persists_the_label_and_its_photo_row_on_a_real_sqlite_store(
    tmp_path, feature_set
):
    store = SQLiteStore(tmp_path / "store.db")
    embedder = _Embedder()
    decider = RankerDecider(
        _AllPhotos(), embedder, PreferenceModel(40), feature_set=feature_set
    )
    try:
        result = _run_training(store, decider, "like")
        assert result["status"] == "completed", result

        # The training driver's profile is one frame named "Ari".
        v1 = _v1_vector("Ari")
        ((run_id, app, label_pid, source),) = _sqlite_rows(
            store, "SELECT run_id, app, profile_id, source FROM labels"
        )
        assert (run_id, app, source) == ("training-run", "hinge", "manual")
        # labels.embedding is the v1 vector in every mode (invariant 2).
        assert store.load_label_rows() == [(True, v1, "manual")]

        (row,) = _photo_table(store)
        expected = _photo_vector([_FRAME])
        assert row == (
            "training-run",
            "hinge",
            label_pid,  # the label's own profile_id: the join key
            VERSION,
            json.dumps(expected),
            1,
            frames_sha256([_FRAME]),
            "live",
        )
        assert embedder.whole_calls == [[_FRAME]]  # ALL frames, computed once

        assert store.load_label_feature_rows(VERSION) == [
            (True, v1, "manual", expected, True)
        ]
        assert store.photo_feature_coverage(VERSION) == {
            "labels": 1,
            "with_features": 1,
            "missing": 0,
            "zero_cards": 0,
        }
        assert store.labels_missing_photo_features(VERSION) == []

        assert fetch_label_rows(store, V1) == [(True, v1, "manual")]
        assert fetch_label_rows(store, WHOLE) == [(True, v1 + expected, "manual")]
    finally:
        store.close()


# --------------------------------------------------------------------------------------------
# AUTO: an archived pass gets its photo row; its review answer joins it.
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("feature_set", [V1, WHOLE])
def test_an_archived_auto_pass_row_joins_its_review_label(tmp_path, feature_set):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        _seed_labels(store, 24)
        embedder = _Embedder()
        model = PreferenceModel(10, min_per_class=5)
        decider = RankerDecider(_AllPhotos(), embedder, model, feature_set=feature_set)
        assert decider.retrain(store) is True
        expected_dims = V1_DIMS if feature_set == V1 else V1_DIMS + WHOLE_PHOTO_DIMS
        assert model._feature_dim == expected_dims

        profile = Profile(photos=_frames_for("Pauto", 2), name="Pauto")
        events = []
        worker = Worker(
            "bumble",
            _AutoDriver([profile], events),
            decider,
            None,
            store,
            "auto-run",
            _Pacing(),
            threading.Event(),
            mode="auto",
            auto_review_random_share=1.0,
        )
        worker.run()
        assert "dislike" in events

        ((pass_pid, pass_embedding),) = _sqlite_rows(
            store, "SELECT profile_id, embedding FROM auto_passes"
        )
        # auto_passes.embedding is the v1 vector in every mode (invariant 2).
        assert json.loads(pass_embedding) == _v1_vector("Pauto")
        photos = list(profile.photos)
        pass_rows = [r for r in _photo_table(store) if r[2] == pass_pid]
        assert pass_rows == [
            (
                "auto-run",
                "bumble",
                pass_pid,
                VERSION,
                json.dumps(_photo_vector(photos)),
                2,
                frames_sha256(photos),
                "live",
            )
        ]
        # Computed exactly once: in decide() (whole-photo mode) or in the archive (v1 mode).
        assert embedder.whole_calls == [photos]
        assert len(store.load_label_rows()) == 24  # an unreviewed pass is never a label

        store.record_auto_pass_review(profile_id=pass_pid, would_like=True)
        rows = store.load_label_feature_rows(VERSION)
        assert len(rows) == 25
        (answer,) = [r for r in rows if r[2].startswith("auto_review")]
        assert answer == (
            True,
            _v1_vector("Pauto"),
            answer[2],
            _photo_vector(photos),
            True,
        )
        assert store.labels_missing_photo_features(VERSION) == []
        whole_rows = fetch_label_rows(store, WHOLE)
        assert [len(r[1]) for r in whole_rows] == [V1_DIMS + WHOLE_PHOTO_DIMS] * 25
    finally:
        store.close()


# --------------------------------------------------------------------------------------------
# Loader -> model: 2048-d rows train, identity groups read dims 0-511, a gap is NOT READY.
# --------------------------------------------------------------------------------------------
def test_preference_model_trains_on_the_loaded_2048_dim_rows(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        _seed_labels(store, 30)
        v1_rows = fetch_label_rows(store, V1)
        wp_rows = fetch_label_rows(store, WHOLE)
        assert [r[1][:V1_DIMS] for r in wp_rows] == [r[1] for r in v1_rows]
        assert {len(r[1]) for r in wp_rows} == {V1_DIMS + WHOLE_PHOTO_DIMS}
        # Both feature sets group people identically: one person per seeded name.
        assert len(set(identity_groups([r[1][:512] for r in wp_rows]))) == 30

        model = PreferenceModel(10, min_per_class=5)
        assert model.train(wp_rows) is True
        assert model._feature_dim == V1_DIMS + WHOLE_PHOTO_DIMS
        like = _v1_vector("Lprobe") + _photo_vector(_frames_for("Lprobe"))
        decision, score = model.decide(like)
        assert decision == "like" and 0.0 <= score <= 1.0
    finally:
        store.close()


def test_a_label_without_a_photo_row_leaves_the_ranker_not_ready_everywhere(
    tmp_path, capsys
):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        _seed_labels(store, 12)
        store.add_label(
            "r", "hinge", False, _v1_vector("Pgap"), source="manual", profile_id="gap"
        )
        with pytest.raises(FeatureSetIncomplete) as caught:
            fetch_label_rows(store, WHOLE)
        reason = f"1 of 13 labels lack whole-photo features; run {BACKFILL}"
        assert caught.value.reason == reason
        with pytest.raises(FeatureSetIncomplete):
            fetch_training_inputs(store, WHOLE)
        assert len(fetch_label_rows(store, V1)) == 13  # v1 never reads the new table

        model = PreferenceModel(10, min_per_class=5)
        decider = RankerDecider(_AllPhotos(), _Embedder(), model, feature_set=WHOLE)
        assert RankerDecider(_AllPhotos(), _Embedder(), model).retrain(store) is True
        assert model.ready
        assert decider.retrain(store) is False
        assert (model.ready, model.not_ready_reason, model.n_labels) == (
            False,
            reason,
            13,
        )
        # The worker's AUTO loop defers on a model that is not ready.
        assert decider.decide(Profile(photos=[b"x"], name="Lx")).decision == "defer"

        with pytest.raises(FeatureSetIncomplete):
            HubState._eval_label_pairs(store, WHOLE)
    finally:
        store.close()


def test_hub_eval_pairs_are_manual_only_and_whole_photo_shaped(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        _seed_labels(store, 6)
        pairs = HubState._eval_label_pairs(store, WHOLE)
        assert len(pairs) == 6
        assert {len(vec) for _liked, vec in pairs} == {V1_DIMS + WHOLE_PHOTO_DIMS}
        assert [vec[:V1_DIMS] for _l, vec in pairs] == [
            vec for _l, vec in HubState._eval_label_pairs(store, V1)
        ]
    finally:
        store.close()


def test_hub_model_quality_card_reads_the_configured_feature_set_from_a_real_store(
    tmp_path,
):
    path = _write_config(
        tmp_path,
        ranker={"min_labels_to_engage": 10, "min_per_class": 5, "feature_set": WHOLE},
    )
    store = SQLiteStore(tmp_path / "store.db")
    _seed_labels(store, 20, app="bumble")
    store.add_label(
        "r", "bumble", True, _v1_vector("Lgap"), source="manual", profile_id="gap"
    )
    store.close()

    snapshot = HubState(str(path)).eval_snapshot()

    assert snapshot["status"] == "incomplete_features", snapshot
    assert (
        f"1 of 21 labels lack whole-photo features; run {BACKFILL}"
        in snapshot["message"]
    )
    assert (
        "whole-photo rows for 20 of 21 labels (1 missing, 0 with zero photo cards)"
        in snapshot["message"]
    )


# --------------------------------------------------------------------------------------------
# Startup and stats read the configured feature set from config.yaml.
# --------------------------------------------------------------------------------------------
def test_supervisor_startup_trains_whole_photo_rows_from_a_real_store(
    monkeypatch, tmp_path, capsys
):
    import operation_love.supervisor as sup

    built = []
    real_decider = sup.RankerDecider

    class SpyDecider(real_decider):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            built.append(self)

    monkeypatch.setattr(sup, "RankerDecider", SpyDecider)
    store = SQLiteStore(tmp_path / "store.db")
    _seed_labels(store, 20)
    spy, _snap = _run_supervisor(
        monkeypatch, tmp_path, store, min_labels=10, extra=f"  feature_set: {WHOLE}"
    )
    out = capsys.readouterr().out
    trained = spy.models[0]["trained"][0][0]
    assert {len(r[1]) for r in trained} == {V1_DIMS + WHOLE_PHOTO_DIMS}
    assert "Ranker: labels=20 ready=True" in out
    assert f"Ranker feature set: {WHOLE}" in out
    # The worker scores with the model that was trained on those rows, in the same mode.
    (decider,) = built
    assert decider.feature_set == WHOLE and decider.model.ready
    assert decider.model._feature_dim == V1_DIMS + WHOLE_PHOTO_DIMS
    assert spy.workers  # a worker was built around it


def test_supervisor_startup_reports_the_backfill_command_for_a_missing_row(
    monkeypatch, tmp_path, capsys
):
    store = SQLiteStore(tmp_path / "store.db")
    _seed_labels(store, 20)
    store.add_label(
        "r", "hinge", True, _v1_vector("Lgap"), source="manual", profile_id="gap"
    )
    spy, snap = _run_supervisor(
        monkeypatch, tmp_path, store, min_labels=10, extra=f"  feature_set: {WHOLE}"
    )
    out = capsys.readouterr().out
    assert spy.models[0]["trained"] == []  # never trained on a subset
    assert "Ranker: labels=21 ready=False" in out
    assert (
        f"Ranker not ready: 1 of 21 labels lack whole-photo features; run {BACKFILL}"
        in out
    )


def test_stats_reads_the_feature_set_and_prints_coverage(tmp_path, capsys):
    path = _write_config(
        tmp_path,
        ranker={"min_labels_to_engage": 10, "min_per_class": 5, "feature_set": WHOLE},
    )
    store = SQLiteStore(tmp_path / "store.db")
    _seed_labels(store, 20, app="bumble")
    store.close()

    stats.show(str(path))
    out = capsys.readouterr().out
    assert "Labels  : 20  (liked 10 / passed 10)" in out
    assert f"Features: {WHOLE}" in out
    assert (
        "Features: whole-photo rows for 20 of 20 labels (0 missing, 0 with zero photo cards)"
        in out
    )
    assert "Ranker  : ready=True" in out

    store = SQLiteStore(tmp_path / "store.db")
    store.add_label(
        "r", "bumble", True, _v1_vector("Lgap"), source="manual", profile_id="gap"
    )
    store.close()
    stats.show(str(path))
    out = capsys.readouterr().out
    assert "Labels  : 21  (liked 11 / passed 10)" in out
    assert "(1 missing, 0 with zero photo cards)" in out
    assert "Ranker  : ready=False" in out
    assert f"Not ready: 1 of 21 labels lack whole-photo features; run {BACKFILL}" in out


# --------------------------------------------------------------------------------------------
# Backfill.
# --------------------------------------------------------------------------------------------
class _NotFound(Exception):
    pass


class _Blob:
    def __init__(self, objects, name):
        self._objects, self.name = objects, name

    def upload_from_string(self, data, content_type=None, timeout=None, retry=None):
        self._objects[self.name] = bytes(data)

    def download_as_bytes(self, timeout=None):
        if self.name not in self._objects:
            raise _NotFound(self.name)
        return self._objects[self.name]


class _GcsBucket(_Bucket):
    def __init__(self, objects):
        super().__init__()
        self._objects = objects

    def blob(self, name):
        return _Blob(self._objects, name)


class _Gcs:
    """One private bucket of in-memory objects (uploads and downloads share it)."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.downloads: list[str] = []

    def bucket(self, _name):
        gcs = self

        class _Tracked(_GcsBucket):
            def blob(self, name):
                blob = super().blob(name)
                original = blob.download_as_bytes

                def download(timeout=None):
                    gcs.downloads.append(name)
                    return original(timeout=timeout)

                blob.download_as_bytes = download
                return blob

        return _Tracked(self.objects)


def _bq(client, gcs, *, ensure=True):
    return BigQueryStore(
        "proj",
        "ds",
        photo_bucket="photos",
        flush_every=100,
        client=client,
        storage_client=gcs,
        ensure=ensure,
    )


def test_backfill_on_sqlite_refuses_labels_whose_frames_it_cannot_supply(
    tmp_path, monkeypatch, capsys
):
    path = _write_config(tmp_path)
    store = SQLiteStore(tmp_path / "store.db")
    _seed_labels(store, 2)
    store.add_label(
        "r", "bumble", True, _v1_vector("Lgap"), source="manual", profile_id="gap"
    )
    store.close()
    embedders = []

    def fake_embedder():
        embedders.append(_Embedder())
        return embedders[-1]

    monkeypatch.setattr(embed_module, "Embedder", fake_embedder)

    assert backfill_tool.main(["--config", str(path)]) == 1
    out = capsys.readouterr().out
    assert "frames unavailable x1" in out and "Backfill INCOMPLETE" in out
    assert embedders[0].whole_calls == []  # nothing was measured, so nothing is written
    store = SQLiteStore(tmp_path / "store.db")
    try:
        assert len(_photo_table(store)) == 2
        assert store.photo_feature_coverage(VERSION)["missing"] == 1
    finally:
        store.close()


def test_backfill_cli_on_a_complete_sqlite_store_has_nothing_to_do(
    tmp_path, monkeypatch, capsys
):
    path = _write_config(tmp_path)
    store = SQLiteStore(tmp_path / "store.db")
    _seed_labels(store, 3)
    store.close()
    embedders = []
    monkeypatch.setattr(
        embed_module, "Embedder", lambda: embedders.append(_Embedder()) or embedders[-1]
    )

    assert backfill_tool.main(["--config", str(path)]) == 0
    out = capsys.readouterr().out
    assert "0 label(s) missing a row" in out and "Backfill complete." in out
    assert embedders[0].warmed == 0  # no model load when there is no work


def test_backfill_fills_a_training_label_identically_to_its_live_row(tmp_path):
    """Live Training row -> lost -> backfilled from the archived GCS frames: same row.

    Uses the real BigQueryStore (its SQL runs on SQLite) so the frames really travel through
    record_profile's upload, profile_photos, labels_missing_photo_features and FrameSource.
    """
    client, gcs = _SqlBQ(), _Gcs()
    store = _bq(client, gcs)
    embedder = _Embedder()
    decider = RankerDecider(
        _AllPhotos(), embedder, PreferenceModel(40), feature_set=WHOLE
    )
    result = _run_training(store, decider, "dislike")
    assert result["status"] == "completed", result
    store.close()

    (live,) = client.rows("photo_features")
    (label,) = client.rows("labels")
    assert live["profile_id"] == label["profile_id"] and live["source"] == "live"
    photos = client.rows("profile_photos")
    assert [p["photo_index"] for p in photos] == [0]
    assert gcs.objects[photos[0]["gcs_uri"].split("gs://photos/", 1)[1]] == _FRAME

    # A label from before the feature existed: no row.
    client.con.execute('DELETE FROM "proj.ds.photo_features"')
    store = _bq(client, gcs)
    try:
        with pytest.raises(FeatureSetIncomplete):
            fetch_label_rows(store, WHOLE)
        (entry,) = store.labels_missing_photo_features(VERSION)
        assert entry["frame_sha256s"] == [hashlib.sha256(_FRAME).hexdigest()]

        backfill_embedder = _Embedder()
        report = backfill_tool.backfill(
            store,
            backfill_embedder,
            backfill_tool.FrameSource(store).frames,
            feature_version=VERSION,
            dims=WHOLE_PHOTO_DIMS,
            log=lambda _line: None,
        )
        assert report.ok and report.written == 1, report.failures
        assert backfill_embedder.whole_calls == [[_FRAME]]
        assert store.load_label_feature_rows(VERSION)[0][4] is True
        assert len(fetch_label_rows(store, WHOLE)[0][1]) == V1_DIMS + WHOLE_PHOTO_DIMS
    finally:
        store.close()

    (filled,) = client.rows("photo_features")
    for key in ("run_id", "app", "profile_id", "feature_version", "embedding"):
        assert filled[key] == live[key], key
    assert filled["card_count"] == live["card_count"]
    assert filled["frames_sha256"] == live["frames_sha256"]
    assert filled["source"] == "backfill"

    # Idempotent: a fresh process finds nothing left to do and loads no model.
    store = _bq(client, gcs)
    try:
        again = _Embedder()
        report = backfill_tool.backfill(
            store,
            again,
            backfill_tool.FrameSource(store).frames,
            feature_version=VERSION,
            dims=WHOLE_PHOTO_DIMS,
            log=lambda _line: None,
        )
        assert (report.listed, report.written, again.warmed) == (0, 0, 0)
    finally:
        store.close()


# --------------------------------------------------------------------------------------------
# compare_feature_sets on a real store.
# --------------------------------------------------------------------------------------------
def test_compare_feature_sets_runs_on_a_real_sqlite_store(tmp_path, capsys):
    path = _write_config(tmp_path, ranker={"like_recall_target": 0.9})
    store = SQLiteStore(tmp_path / "store.db")
    ids = _seed_labels(store, 40, app="bumble")
    store.close()

    code = compare_tool.main(["--config", str(path), "--seeds", "2", "--orders", "2"])
    out = capsys.readouterr().out
    assert code in (compare_tool.EXIT_SWITCH, compare_tool.EXIT_KEEP), out
    assert f"{V1} ({V1_DIMS}-d) vs {WHOLE} ({V1_DIMS + WHOLE_PHOTO_DIMS}-d)" in out
    assert "Labels: 40 (20 like / 20 pass; 40 manual, 0 auto-review answers)" in out
    assert "unavailable" not in out  # time-forward read the real store's timestamps
    assert "VERDICT:" in out
    assert not any(pid in out for pid in ids)  # aggregates only

    store = SQLiteStore(tmp_path / "store.db")
    with store._lock:
        store.con.execute("DELETE FROM photo_features WHERE profile_id = ?", (ids[0],))
        store.con.commit()
    store.close()
    code = compare_tool.main(["--config", str(path), "--seeds", "2", "--orders", "2"])
    out = capsys.readouterr().out
    assert code == compare_tool.EXIT_REFUSED
    assert f"1 of 40 labels lack whole-photo features; run {BACKFILL}" in out
