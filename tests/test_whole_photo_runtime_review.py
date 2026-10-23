"""Runtime-safety review of the whole-photo feature (2026-10-23).

1. Default ``face_crop_v1`` mode.  Contract invariant 5: "Training never fails because the new feature failed in v1 mode (log
loudly, keep the label; the backfill can fill it later)."  The worker honours that for the
two failure points it can see -- computing the vector and ``add_photo_features`` -- but on
BigQuery ``add_photo_features`` only BUFFERS the row.  The insert happens in the Training
persist's ``store.flush()``, which flushes every table and raises if any one of them fails,
so a ``photo_features`` insert failure fails the Hub action (and ends the Training run) in v1
mode even though the label itself landed.  The same buffered row then fails the supervisor's
shutdown flush, which marks the whole run ``save_failed``.

2. ``face_crop_whole_photo_v1`` mode.  Frames in which the extractor finds no photo card at all
(a Hinge layout it no longer recognises) came back as a zero-card profile, which was scored
with a 768-d zero block exactly like a "recorded" zero-card fact.  A Hinge profile always
carries photos, so on Hinge that is an extraction failure being scored as v1-plus-zeros: the
silent fallback invariant 4 forbids.  (Blank or loading frames have no face, so in the real path
``decide()`` returns no_face before the whole-photo vector; the test stubs the face vector.)

3. The backfill's per-label error handling does not cover a store write that raises (the
shipped ``storage.bigquery.flush_every: 1`` makes every BigQuery ``add_photo_features`` a
synchronous insert), so one such failure aborts the whole run instead of being reported.

Synthetic data only; the BigQuery client is the fake from tests/test_bigquery_store.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from operation_love.perception.capture import Profile
from operation_love.ranker.bigquery_store import BigQueryStore
from operation_love.ranker.decider import RankerDecider
from operation_love.ranker.threshold import FEATURE_SET_FACE_CROP_V1
from operation_love.vision.photo_cards import (
    WHOLE_PHOTO_DIMS,
    WHOLE_PHOTO_FEATURE_VERSION,
)
from tests.test_bigquery_store import _FakeBQ, _FakeStorage
from tests.test_training_worker import (
    _TIMEOUT_S,
    _checkpoint,
    _new_worker,
    _submit,
    _wait_until,
)
from tests.test_whole_photo_wiring import WHOLE, _Model, _PhotoDecider, _Quality

SHA = "ab" * 32


@pytest.fixture(autouse=True)
def _stub_native_training_notification(monkeypatch):
    """Worker tests must never display real host notifications (or play the alert sound)."""
    import operation_love.worker as worker_module

    monkeypatch.setattr(worker_module, "notify_training_decision_ready", lambda: True)


class _PhotoFeaturesRejectingBQ(_FakeBQ):
    """Every table inserts normally except ``photo_features``, which BigQuery rejects."""

    def __init__(self, *, raise_exc: Exception | None = None):
        super().__init__()
        self.raise_exc = raise_exc
        self.photo_attempts = 0

    def insert_rows_json(self, table_id, rows, row_ids=None):
        if table_id.endswith(".photo_features"):
            self.photo_attempts += 1
            if self.raise_exc is not None:
                raise self.raise_exc
            return [{"index": 0, "errors": [{"reason": "backendError"}]}]
        return super().insert_rows_json(table_id, rows, row_ids=row_ids)


def _bq(client, flush_every=25) -> BigQueryStore:
    return BigQueryStore(
        "proj",
        "ds",
        photo_bucket="photos",
        flush_every=flush_every,
        client=client,
        storage_client=_FakeStorage(),
        ensure=False,
    )


def _v1_training_run_with_bigquery_photo_rows(client, flush_every):
    """One v1 Training Like whose photo row and flush go through a real BigQueryStore."""
    worker, driver, decider, opener, store, bridge, events = _new_worker()
    worker.decider = _PhotoDecider(decider, feature_set=FEATURE_SET_FACE_CROP_V1)
    bq = _bq(client, flush_every)
    # The harness store keeps every other table; only the new row is BigQuery's, and the
    # persist's flush also flushes BigQuery's buffers, exactly as BigQueryStore.flush does
    # for every table it holds.
    store.add_photo_features = bq.add_photo_features
    harness_flush = store.flush

    def flush():
        harness_flush()
        bq.flush()

    store.flush = flush
    worker.start()
    card = _wait_until(lambda: _checkpoint(bridge))
    _submit(bridge, card, "like")
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive()
    result = bridge.snapshot(run_id="training-run", app="hinge")["results"][-1]
    return SimpleNamespace(result=result, store=store, bq=bq, events=events)


# flush_every=1 is the shipped config.yaml value: add_photo_features then inserts at once,
# raises, and the worker's WARNING path catches it -- but the row stays buffered and the
# persist's flush() resends it and raises anyway.
@pytest.mark.parametrize(
    "flush_every", [1, 25], ids=["shipped-flush_every-1", "batched"]
)
def test_v1_training_is_not_failed_by_a_rejected_photo_features_insert(
    flush_every, capsys
):
    """FAILS today: the label landed, but the Hub action is reported failed and the run halts.

    In v1 mode the whole-photo row is best-effort (invariant 5), so a BigQuery rejection of
    that row alone must leave the Training action "completed" with a loud WARNING.
    """
    client = _PhotoFeaturesRejectingBQ()
    run = _v1_training_run_with_bigquery_photo_rows(client, flush_every)

    # The v1 label itself was written by the persist before the flush.
    assert len(run.store.labels) == 1
    # The photo_features insert was attempted and refused.
    assert client.photo_attempts >= 1
    assert run.result["status"] == "completed", (
        "a best-effort photo_features row failed the v1 Training action: "
        f"{run.result}\n{capsys.readouterr().out[-2000:]}"
    )


def test_a_raised_photo_features_insert_is_never_dropped_so_every_later_flush_fails():
    """Mechanism (passes): an insert that RAISES (not row errors) is not fail-counted, so the
    buffered photo_features row is resent and fails on every flush, including close().

    In v1 mode that is: every later Training flush of the run and the supervisor's shutdown
    flush (``phase = "save_failed"``, every app ``error``) fail because of a row nothing in v1
    reads, while the labels table landed on the first attempt.
    """

    class NotFound(Exception):
        pass

    client = _PhotoFeaturesRejectingBQ(
        raise_exc=NotFound("404 Not found: Table proj:ds.photo_features")
    )
    store = _bq(client)
    store.add_label("r1", "hinge", True, [0.1, 0.2], source="manual", profile_id="p1")
    assert store.add_photo_features(
        run_id="r1",
        app="hinge",
        profile_id="p1",
        feature_version=WHOLE_PHOTO_FEATURE_VERSION,
        embedding=[0.01] * WHOLE_PHOTO_DIMS,
        card_count=3,
        frames_sha256=SHA,
        source="live",
    )
    for _ in range(7):
        with pytest.raises(RuntimeError, match="photo_features"):
            store.flush()
    assert len(client.inserted["proj.ds.labels"]) == 1  # the label landed first time
    assert client.photo_attempts == 7  # never dropped: resent and failed on every flush
    with pytest.raises(RuntimeError, match="photo_features"):
        store.close()


# --- 2. whole-photo mode: a capture with no photo card is scored, not refused -----------------
def _plain_frame_png(value: int) -> bytes:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    ok, buf = cv2.imencode(".png", np.full((2400, 1080, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


@pytest.mark.parametrize("value", [245, 0], ids=["blank-white", "black"])
def test_whole_photo_mode_refuses_to_score_a_hinge_capture_with_no_photo_card(
    value, monkeypatch
):
    """Before the fix the real extractor + the real embed_whole_photos returned a zero-card
    embedding (no error anywhere) for 12 frames holding no photo card, and decide() scored
    v1 ++ 768 zeros.  Now the embedder raises ``NoPhotoCards`` and the decision stops.

    Only the model load is skipped (zero cards never reach CLIP); extraction,
    ``Embedder.embed_whole_photos`` and the decider are the shipped code.
    """
    from operation_love.vision.embed import Embedder
    from operation_love.vision.photo_cards import NoPhotoCards

    embedder = Embedder()
    monkeypatch.setattr(embedder, "_ensure", lambda: None)
    monkeypatch.setattr(embedder, "embed_profile", lambda profile: [0.5, 0.5])
    model = _Model(ready=True, decision=("like", 0.9))
    decider = RankerDecider(_Quality(), embedder, model, feature_set=WHOLE)
    profile = Profile(photos=[_plain_frame_png(value)] * 12, name="Avery")

    with pytest.raises(NoPhotoCards, match="12 frame"):
        embedder.embed_whole_photos(profile.photos)

    try:
        decision = decider.decide(profile)
    except NoPhotoCards:  # the fail-loud outcome this test asks for
        assert model.seen == []  # nothing was scored
        return
    pytest.fail(
        f"whole-photo mode scored a capture with no photo card as {decision.decision!r} "
        f"(score {decision.score}) on v1 ++ {len(model.seen[0]) - 2} zeros"
    )


# --- 3. the backfill: a raising store write aborts the run ---------------------------------
def test_backfill_reports_a_raising_store_write_as_that_labels_failure_and_continues():
    """FAILS today: ``backfill()`` promises "Never raises per label", but add_photo_features
    raising (a BigQuery insert error at flush_every 1) propagates out of the loop, so the
    remaining labels are not written and no summary or coverage is printed."""
    from tests.test_backfill_photo_features import _run, _world

    store, _objects = _world(3)
    real_add = store.add_photo_features

    def add_photo_features(**row):
        if row["profile_id"] == "p1":
            raise RuntimeError("BigQuery insert errors for photo_features: [...]")
        return real_add(**row)

    store.add_photo_features = add_photo_features
    try:
        report, _lines = _run(store, workers=1)
    except RuntimeError as exc:
        pytest.fail(f"backfill() raised instead of reporting the label: {exc}")
    assert not report.ok
    assert [name for name, _category, _reason in report.failures] == [
        "hinge/p1 (run run1)"
    ]
    assert {w["profile_id"] for w in store.writes} == {"p0", "p2"}
