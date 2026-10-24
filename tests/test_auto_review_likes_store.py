"""Archived AUTO LIKES in the review store, in BOTH backends.

AUTO now archives a landed model like into the same `auto_passes` table as a model pass, with
`decision = "like"`, so the owner can give their own Like/Dislike on it. What these tests pin:

* a like is never part of the `random` / `near_cut` pass pools, their counts, the false-pass
  statistics (`random_*`, `pass_share` inputs) or the archive window -- the live correction
  reads exactly what it read before likes were archived;
* the `likes` pool is uniform over unreviewed archived likes; its answers are labels with
  source `auto_review_like` (train-only) and can be undone;
* `auto_review_waiting` counts exactly what each pool can still offer;
* retention bounds likes and passes together;
* a row written before the `decision` column existed reads as a pass, on SQLite and on a
  read-only BigQuery store whose table has not been migrated yet.

The BigQuery half runs the store's real SQL through test_auto_review_store's `_SqlBQ` double.
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

import operation_love.ranker.bigquery_store as bq_module
from operation_love.ranker import Store
from operation_love.ranker.bigquery_store import _TABLES, BigQueryStore
from operation_love.ranker.store import (
    SQLiteStore,
    missing_photo_feature_entries,
)
from operation_love.ranker.threshold import is_calibration_source, live_adjustment
from operation_love.vision.photo_cards import WHOLE_PHOTO_FEATURE_VERSION
from tests.test_auto_review_store import (  # noqa: F401 - the autouse SDK stub applies here too
    _archive,
    _bigquery_sdk_stub,
    _bq_store,
    _columns,
    _Harness,
    _HookConnection,
    _Job,
    _png,
    _SqlBQ,
    _Storage,
)

_PASS_KEYS = (
    "pending_random",
    "pending_near_cut",
    "random_reviewed",
    "random_would_like",
    "near_cut_reviewed",
    "near_cut_would_like",
    "auto_likes",
    "auto_passes",
    "auto_model_passes",
)


@pytest.fixture(params=["sqlite", "bigquery"])
def h(request, tmp_path):
    harness = _Harness(request.param, tmp_path)
    yield harness
    try:
        harness.close()
    except Exception:  # noqa: BLE001 - a test may leave a deliberately failing buffer
        pass


def _like(h, pid, **overrides):
    return _archive(h, pid, decision="like", **overrides)


def _manifest(h) -> dict[str, tuple[object, bool]]:
    """profile_id -> (stored decision, stored random_sample) for every archived row."""
    if h.kind == "sqlite":
        rows = h.store.con.execute(
            "SELECT profile_id, decision, random_sample FROM auto_passes"
        ).fetchall()
    else:
        h.store.flush()
        rows = h.client.con.execute(
            'SELECT profile_id, decision, random_sample FROM "proj.ds.auto_passes"'
        ).fetchall()
    return {pid: (decision, bool(flag)) for pid, decision, flag in rows}


def _set_decision(h, pid, value):
    """Overwrite one archived row's stored decision (NULL = a pre-column row)."""
    if h.kind == "sqlite":
        h.store.con.execute(
            "UPDATE auto_passes SET decision=? WHERE profile_id=?", (value, pid)
        )
        h.store.con.commit()
    else:
        h.store.flush()
        h.client.con.execute(
            'UPDATE "proj.ds.auto_passes" SET decision=? WHERE profile_id=?',
            (value, pid),
        )


def _drain(h, kind, n=150):
    return {
        h.store.next_auto_pass_for_review(kind=kind)["profile_id"] for _ in range(n)
    }


# ------------------------------------------------------------------ surface
def test_both_stores_expose_auto_review_waiting_and_the_decision_keyword_like_the_protocol():
    for name in ("auto_review_waiting", "record_auto_pass"):
        protocol = inspect.signature(getattr(Store, name))
        for cls in (SQLiteStore, BigQueryStore):
            actual = inspect.signature(getattr(cls, name))
            assert list(actual.parameters) == list(protocol.parameters), (cls, name)
            for pname, param in list(actual.parameters.items())[1:]:
                assert param.kind is inspect.Parameter.KEYWORD_ONLY, (cls, name, pname)
                assert param.default == protocol.parameters[pname].default, (
                    cls,
                    name,
                    pname,
                )
    assert inspect.signature(Store.record_auto_pass).parameters["decision"].default == (
        "dislike"
    )


# ------------------------------------------------------------------ archiving a like
def test_a_like_is_archived_with_its_decision_and_never_in_the_random_sample(h):
    assert _like(h, "l1", random_sample=True, name="Ada") is True
    assert _archive(h, "p1", random_sample=True) is True
    assert _manifest(h) == {"l1": ("like", False), "p1": ("dislike", True)}
    # Archiving is not a label or a decision, for a like exactly as for a pass.
    assert h.count("labels") == 0
    assert h.count("decisions") == 0
    assert h.count("auto_pass_photos") == 4


@pytest.mark.parametrize(
    "decision",
    ["LIKE", "Like", "pass", "", None, 1, True, b"like"],
    ids=["upper", "title", "pass", "empty", "none", "int", "bool", "bytes"],
)
def test_a_malformed_decision_is_refused_before_anything_is_stored(h, decision):
    with pytest.raises(ValueError, match="decision"):
        _archive(h, "x", decision=decision)
    assert h.count("auto_passes") == 0
    assert h.count("auto_pass_photos") == 0
    if h.kind == "bigquery":
        assert h.storage.bucket("photos").upload_log == []


def test_a_like_still_needs_a_real_random_sample_bool(h):
    with pytest.raises(ValueError, match="random_sample"):
        _like(h, "l1", random_sample=None)
    assert h.count("auto_passes") == 0


# ------------------------------------------------------------------ pools
def test_likes_never_enter_the_pass_pools_and_the_likes_pool_holds_only_likes(h):
    _archive(h, "r1", random_sample=True)
    _archive(h, "r2", random_sample=True)
    _archive(h, "n1", score=0.30, threshold=0.5)
    _archive(h, "n2", score=0.20, threshold=0.5)
    _like(h, "l1", random_sample=True)  # stored False: never the unbiased pass sample
    _like(h, "l2", score=0.499, threshold=0.5)  # the "closest to the cut" row of all
    _like(h, "l3", score=0.95)

    assert _drain(h, "random") == {"r1", "r2"}
    assert _drain(h, "likes") == {"l1", "l2", "l3"}  # uniform: every like reachable
    near = []
    while (item := h.store.next_auto_pass_for_review(kind="near_cut")) is not None:
        near.append(item["profile_id"])
        assert item["kind"] == "near_cut"
        h.store.record_auto_pass_review(profile_id=item["profile_id"], would_like=False)
    assert near == ["n1", "n2"]
    item = h.store.next_auto_pass_for_review(kind="likes")
    assert item["kind"] == "likes"
    assert set(item) == {
        "profile_id",
        "app",
        "run_id",
        "created_at",
        "score",
        "threshold",
        "photo_count",
        "profile_name",
        "kind",
    }


def test_the_likes_pool_is_not_ordered_by_score(h):
    for i in range(4):
        _like(h, f"l{i}", score=0.6 + i / 10)
    firsts = {
        h.store.next_auto_pass_for_review(kind="likes")["profile_id"] for _ in range(80)
    }
    assert firsts == {"l0", "l1", "l2", "l3"}


def test_the_likes_pool_filters_by_app_and_drops_answered_likes(h):
    _like(h, "hin", app="hinge")
    _like(h, "bum", app="bumble")
    _like(h, "hin2", app="hinge")
    h.store.record_auto_pass_review(profile_id="hin2", would_like=True)
    for _ in range(30):
        assert (
            h.store.next_auto_pass_for_review(kind="likes", app="bumble")["profile_id"]
            == "bum"
        )
        assert (
            h.store.next_auto_pass_for_review(kind="likes", app="hinge")["profile_id"]
            == "hin"
        )
    assert h.store.next_auto_pass_for_review(kind="likes", app="tinder") is None


# ------------------------------------------------------------------ answering a like
def test_a_like_answer_is_a_train_only_label_with_the_like_source(h):
    emb = [0.125, -0.75, 3.5]
    _like(
        h,
        "l1",
        embedding=emb,
        name="Ada",
        photos=3,
        run_id="run-7",
        app="bumble",
    )

    result = h.store.record_auto_pass_review(profile_id="l1", would_like=False)

    assert result == {
        "kind": "likes",
        "label_source": "auto_review_like",
        "decision": "like",
        "score": 0.4,
        "threshold": 0.5,
        "would_like": False,
    }
    (label,) = h.labels()
    assert (label["run_id"], label["app"], label["profile_id"]) == (
        "run-7",
        "bumble",
        "l1",
    )
    assert label["liked"] is False  # the OWNER's answer, never the bot's like
    assert label["source"] == "auto_review_like"
    assert label["embedding"] == emb
    assert label["photo_count"] == 3
    assert is_calibration_source(label["source"]) is False
    (review,) = h.reviews()
    assert (review["profile_id"], review["would_like"], review["kind"]) == (
        "l1",
        False,
        "likes",
    )
    assert h.store.load_label_rows() == [(False, emb, "auto_review_like")]
    for answer in (False, True):
        with pytest.raises(ValueError, match="already reviewed"):
            h.store.record_auto_pass_review(profile_id="l1", would_like=answer)
    assert len(h.labels()) == 1


def test_undo_reopens_a_like_answer_and_it_can_be_answered_again(h):
    _archive(h, "p1", random_sample=True)
    h.store.record_auto_pass_review(profile_id="p1", would_like=True)
    _like(h, "l1", name="Lia")
    h.store.record_auto_pass_review(profile_id="l1", would_like=False)
    assert h.store.auto_review_waiting()["likes"] == 0

    undone = h.store.undo_latest_auto_review()

    assert undone == {
        "profile_id": "l1",
        "profile_name": "Lia",
        "kind": "likes",
        "would_like": False,
    }
    assert [(r["profile_id"], r["source"]) for r in h.labels()] == [
        ("p1", "auto_review_random")
    ]
    assert [r["profile_id"] for r in h.reviews()] == ["p1"]
    assert h.store.auto_review_waiting()["likes"] == 1
    assert h.store.next_auto_pass_for_review(kind="likes")["profile_id"] == "l1"
    again = h.store.record_auto_pass_review(profile_id="l1", would_like=True)
    assert again["label_source"] == "auto_review_like"
    assert h.store.auto_review_stats()["likes"] == {
        "archived": 1,
        "reviewed": 1,
        "agreed": 1,
    }


def test_a_like_answer_is_not_like_evidence_for_the_run_it_came_from(h):
    _like(h, "l1", run_id="run-A")
    h.store.record_auto_pass_review(profile_id="l1", would_like=True)
    h.store.flush()
    rows = h.store.advisory_opener_run_rows("run-A", "hinge")
    assert rows["effective_counts"]["like_labels"] == 0
    assert rows["preference_counts"]["labels"] == 1


def test_a_like_answers_frames_come_from_the_review_archive(h):
    _like(h, "l1", photos=2)
    h.store.record_auto_pass_review(profile_id="l1", would_like=True)
    (entry,) = h.store.labels_missing_photo_features(WHOLE_PHOTO_FEATURE_VERSION)
    assert (entry["profile_id"], entry["source"]) == ("l1", "auto_review_like")
    if h.kind == "bigquery":  # SQLite keeps no object URIs, so it lists no frames
        h.store.flush()
        stored = [
            row["gcs_uri"]
            for row in sorted(
                h.client.rows("auto_pass_photos"), key=lambda r: r["photo_index"]
            )
        ]
        assert entry["frame_uris"] == stored and len(stored) == 2


def test_the_frame_resolver_reads_the_review_archive_for_a_like_answer():
    sha = "a" * 64
    key = ("run", "hinge", "l1")
    entries = missing_photo_feature_entries(
        [
            {
                "run_id": "run",
                "app": "hinge",
                "profile_id": "l1",
                "source": "auto_review_like",
            }
        ],
        {},
        {key: [(1.0, 0, "gs://b/manual.png", sha)]},
        {key: [(1.0, 0, "gs://b/review.png", sha)]},
    )
    assert entries[0]["frame_uris"] == ["gs://b/review.png"]


# ------------------------------------------------------------------ the live correction
def _pass_world(h):
    """Archived passes with their decision rows, two answers, a demoted like, and a LIKE
    decision row older than every archived pass (never in the window)."""
    h.store.record_decision(
        "run",
        "hinge",
        "like",
        0.9,
        source="auto",
        profile_id="l-old",
        created_at=500.0,
    )
    for i, pid in enumerate(("p1", "p2", "p3")):
        h.store.record_decision(
            "run",
            "hinge",
            "dislike",
            0.1,
            source="auto",
            profile_id=pid,
            created_at=1000.0 + i,
        )
        _archive(h, pid, random_sample=bool(i % 2))
    h.store.record_decision(
        "run", "hinge", "like", 0.8, source="auto", profile_id="l1", created_at=1010.0
    )
    h.store.record_decision(  # a like the session demoted to a pass: never archived
        "run",
        "hinge",
        "dislike",
        0.7,
        source="auto",
        profile_id="demoted",
        created_at=1011.0,
    )
    h.store.record_auto_pass_review(profile_id="p2", would_like=True)  # random
    h.store.record_auto_pass_review(profile_id="p1", would_like=False)  # near_cut


def test_archived_and_answered_likes_leave_every_pass_statistic_unchanged(h):
    """The invariant behind the live false-pass correction. Mutations this catches: a pass
    pool or its pending count without the decision filter (l1/l-old would join near_cut), the
    window start reading archived likes (l-old at t=500 would open it and count its decision),
    a like answer counted as a random/near-cut review."""
    _pass_world(h)
    before = h.store.auto_review_stats()
    assert before["likes"] == {"archived": 0, "reviewed": 0, "agreed": 0}
    assert (
        before["auto_likes"],
        before["auto_passes"],
        before["auto_model_passes"],
    ) == (
        1,
        4,
        3,
    )

    _like(h, "l-old", random_sample=True)
    h.set_pass_time("l-old", 500.0)
    _like(h, "l1", score=0.49, threshold=0.5)
    _like(h, "l2", app="bumble")
    h.store.record_auto_pass_review(profile_id="l-old", would_like=True)
    h.store.record_auto_pass_review(profile_id="l1", would_like=False)

    after = h.store.auto_review_stats()
    assert {k: after[k] for k in _PASS_KEYS} == {k: before[k] for k in _PASS_KEYS}
    assert after["likes"] == {"archived": 3, "reviewed": 2, "agreed": 1}
    assert h.store.auto_review_stats(app="hinge")["likes"] == {
        "archived": 2,
        "reviewed": 2,
        "agreed": 1,
    }
    assert h.store.auto_review_stats(app="bumble")["likes"] == {
        "archived": 1,
        "reviewed": 0,
        "agreed": 0,
    }
    for rate in (0.0, 0.05):
        assert live_adjustment(
            after, expected_false_pass_rate=rate, base_rate=0.3
        ) == live_adjustment(before, expected_false_pass_rate=rate, base_rate=0.3)


def test_model_passes_and_the_window_count_only_rows_archived_as_passes(h):
    """Pins the explicit decision filter on the decisions<->archive join: a 'dislike' decision
    row whose lineage id matches an archived LIKE is not an archived pass, so it neither opens
    the window nor counts as a model pass."""
    h.store.record_decision(
        "run", "hinge", "dislike", 0.1, source="auto", profile_id="x", created_at=900.0
    )
    _like(h, "x")
    h.store.record_decision(
        "run", "hinge", "like", 0.9, source="auto", profile_id="y", created_at=950.0
    )
    h.store.record_decision(
        "run", "hinge", "dislike", 0.1, source="auto", profile_id="p", created_at=1000.0
    )
    _archive(h, "p")
    h.store.record_decision(
        "run", "hinge", "dislike", 0.1, source="auto", profile_id="z", created_at=1010.0
    )
    _like(h, "z")  # inside the window, but it is still not an archived model pass
    stats = h.store.auto_review_stats()
    assert (stats["auto_likes"], stats["auto_passes"], stats["auto_model_passes"]) == (
        0,
        2,
        1,
    )


def test_likes_alone_open_no_window(h):
    h.store.record_decision(
        "run", "hinge", "like", 0.9, source="auto", profile_id="l1", created_at=900.0
    )
    _like(h, "l1")
    stats = h.store.auto_review_stats()
    assert (stats["auto_likes"], stats["auto_passes"], stats["auto_model_passes"]) == (
        0,
        0,
        0,
    )
    assert (stats["pending_random"], stats["pending_near_cut"]) == (0, 0)
    assert stats["likes"]["archived"] == 1


# ------------------------------------------------------------------ waiting count
def test_waiting_counts_each_pool_and_the_total(h):
    empty = h.store.auto_review_waiting()
    assert empty == {"likes": 0, "random": 0, "near_cut": 0, "total": 0}
    _archive(h, "r1", random_sample=True)
    _archive(h, "n1")
    _archive(h, "n2")
    _like(h, "l1")
    _like(h, "l2", app="bumble")
    h.store.record_auto_pass_review(profile_id="l1", would_like=True)

    waiting = h.store.auto_review_waiting()
    assert waiting == {"likes": 1, "random": 1, "near_cut": 2, "total": 4}
    assert all(type(v) is int for v in waiting.values())
    assert h.store.auto_review_waiting(app="bumble") == {
        "likes": 1,
        "random": 0,
        "near_cut": 0,
        "total": 1,
    }
    # The count agrees with the stats' pending counts for the pass pools.
    stats = h.store.auto_review_stats()
    assert (waiting["random"], waiting["near_cut"]) == (
        stats["pending_random"],
        stats["pending_near_cut"],
    )
    with pytest.raises(ValueError):
        h.store.auto_review_waiting(app="")


@pytest.mark.parametrize("kind", ["likes", "random", "near_cut"])
@pytest.mark.parametrize(
    "damage", ["missing", "wrong_index", "negative_index", "no_photos", "zero_count"]
)
def test_pools_and_waiting_require_every_expected_archived_photo(h, kind, damage):
    """A manifest without its promised photo sequence never contributes a waiting review.

    Counting distinct stored indices alone is insufficient: replacing index 1 with 99 keeps
    that count intact, while the hub's request for index 1 would still fail.
    """
    decision = "like" if kind == "likes" else "dislike"
    _archive(h, "broken", decision=decision, random_sample=kind == "random")
    _archive(h, "whole", decision=decision, random_sample=kind == "random")
    h.store.flush()
    con = h.store.con if h.kind == "sqlite" else h.client.con
    prefix = "" if h.kind == "sqlite" else "proj.ds."
    photos = f'"{prefix}auto_pass_photos"'
    passes = f'"{prefix}auto_passes"'
    if damage == "missing":
        con.execute(f"DELETE FROM {photos} WHERE profile_id='broken' AND photo_index=1")
    elif damage in ("wrong_index", "negative_index"):
        index = 99 if damage == "wrong_index" else -1
        con.execute(
            f"UPDATE {photos} SET photo_index=? WHERE profile_id='broken' AND photo_index=1",
            (index,),
        )
    elif damage == "no_photos":
        con.execute(f"DELETE FROM {photos} WHERE profile_id='broken'")
    else:
        con.execute(f"UPDATE {passes} SET photo_count=0 WHERE profile_id='broken'")
    con.commit()

    assert h.store.next_auto_pass_for_review(kind=kind)["profile_id"] == "whole"
    waiting = h.store.auto_review_waiting()
    assert waiting == {
        "likes": int(kind == "likes"),
        "random": int(kind == "random"),
        "near_cut": int(kind == "near_cut"),
        "total": 1,
    }
    stats = h.store.auto_review_stats()
    assert stats["pending_random"] == int(kind == "random")
    assert stats["pending_near_cut"] == int(kind == "near_cut")


# ------------------------------------------------------------------ rows from before the column
def test_a_row_with_no_decision_reads_as_a_pass(h):
    _archive(h, "old", random_sample=True)
    _set_decision(h, "old", None)
    assert h.store.next_auto_pass_for_review(kind="likes") is None
    assert h.store.next_auto_pass_for_review(kind="random")["profile_id"] == "old"
    assert h.store.auto_review_waiting() == {
        "likes": 0,
        "random": 1,
        "near_cut": 0,
        "total": 1,
    }
    assert h.store.record_auto_pass_review(profile_id="old", would_like=True) == {
        "kind": "random",
        "label_source": "auto_review_random",
        "decision": "dislike",
        "score": 0.4,
        "threshold": 0.5,
        "would_like": True,
    }


def test_a_damaged_decision_is_in_no_pool_and_cannot_be_answered(h):
    _archive(h, "bad", random_sample=True)
    _set_decision(h, "bad", "maybe")
    for kind in ("likes", "random", "near_cut"):
        assert h.store.next_auto_pass_for_review(kind=kind) is None
    assert h.store.auto_review_waiting()["total"] == 0
    with pytest.raises(ValueError, match="no valid decision"):
        h.store.record_auto_pass_review(profile_id="bad", would_like=True)
    assert h.labels() == [] and h.reviews() == []


# ------------------------------------------------------------------ retention
def test_retention_bounds_likes_and_passes_together_and_keeps_answers(h):
    now = 2_000_000_000.0
    for pid, decision, age in (
        ("l-old", "like", 90),
        ("p-old", "dislike", 80),
        ("l-mid", "like", 3),
        ("p-mid", "dislike", 2),
        ("l-new", "like", 1),
    ):
        _archive(h, pid, decision=decision)
        h.set_pass_time(pid, now - age * 86400.0)
    h.store.record_auto_pass_review(profile_id="l-old", would_like=True)
    labels, reviews = h.labels(), h.reviews()

    assert h.store.prune_auto_passes(max_age_days=30.0, now=now) == 2
    assert set(_manifest(h)) == {"l-mid", "p-mid", "l-new"}
    # One count bound over every archived decision, not one per decision.
    assert h.store.prune_auto_passes(max_passes=2, now=now) == 1
    assert set(_manifest(h)) == {"p-mid", "l-new"}
    assert h.count("auto_pass_photos", "profile_id IN ('l-old','l-mid')") == 0
    assert h.labels() == labels and h.reviews() == reviews
    assert h.store.auto_review_stats()["likes"] == {
        "archived": 1,
        "reviewed": 1,
        "agreed": 1,
    }


# ============================================================================================
# SQLite specifics
# ============================================================================================
def test_sqlite_like_paths_run_only_under_the_store_lock(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        unlocked = []

        def watch(sql):
            if not store._lock.locked():
                unlocked.append(sql)

        store.con = _HookConnection(store.con, before=watch)
        store.record_auto_pass(
            run_id="r",
            app="hinge",
            profile_id="l1",
            score=0.9,
            threshold=0.5,
            embedding=[0.1],
            photos=[_png("a")],
            random_sample=False,
            decision="like",
        )
        store.next_auto_pass_for_review(kind="likes")
        store.auto_review_waiting()
        store.auto_review_waiting(app="hinge")
        store.auto_review_stats()
        store.record_auto_pass_review(profile_id="l1", would_like=True)
        store.undo_latest_auto_review()
        assert unlocked == []
    finally:
        store.close()


def test_sqlite_empty_archived_photo_is_not_reviewable(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    try:
        store.record_auto_pass(
            run_id="r",
            app="hinge",
            profile_id="empty",
            score=0.9,
            threshold=0.5,
            embedding=[0.1],
            photos=[_png("a")],
            random_sample=False,
            decision="like",
        )
        store.con.execute("UPDATE auto_pass_photos SET photo=?", (b"",))
        store.con.commit()
        assert store.auto_review_waiting()["total"] == 0
        assert store.next_auto_pass_for_review(kind="likes") is None
    finally:
        store.close()


# ============================================================================================
# BigQuery specifics
# ============================================================================================
@pytest.fixture
def bq():
    client, storage = _SqlBQ(), _Storage()
    return client, storage, _bq_store(client, storage)


def _bq_like(store, pid, *, photos=2, **kwargs):
    return store.record_auto_pass(
        run_id=kwargs.pop("run_id", "run-1"),
        app=kwargs.pop("app", "hinge"),
        profile_id=pid,
        score=kwargs.pop("score", 0.9),
        threshold=kwargs.pop("threshold", 0.5),
        embedding=kwargs.pop("embedding", [0.1, 0.2]),
        photos=[_png(f"{pid}-{i}") for i in range(photos)],
        random_sample=kwargs.pop("random_sample", False),
        decision=kwargs.pop("decision", "like"),
        **kwargs,
    )


def test_bq_a_like_is_buffered_and_written_with_its_decision(bq):
    client, _storage, store = bq
    assert _bq_like(store, "l1", random_sample=True)
    (row,) = store._buf["auto_passes"]
    assert (row["decision"], row["random_sample"]) == ("like", False)
    store.flush()
    ((table, (written,)),) = [
        (t, rows) for t, rows in client.insert_log if t == "auto_passes"
    ]
    assert (written["decision"], written["random_sample"]) == ("like", False)


def test_bq_an_incomplete_like_is_never_offered_or_counted(bq):
    client, _storage, store = bq
    _bq_like(store, "whole")
    _bq_like(store, "torn", photos=3)
    store.flush()
    client.con.execute(
        "DELETE FROM \"proj.ds.auto_pass_photos\" WHERE profile_id='torn' AND photo_index=2"
    )
    assert {
        store.next_auto_pass_for_review(kind="likes")["profile_id"] for _ in range(60)
    } == {"whole"}
    assert store.auto_review_waiting()["likes"] == 1


def test_bq_like_queries_bind_every_value(bq):
    client, _storage, store = bq
    _bq_like(store, "l1", app="hinge")
    client.queries.clear()
    store.next_auto_pass_for_review(kind="likes", app="hinge")
    store.auto_review_waiting(app="hinge")
    store.auto_review_stats(app="hinge", window=7)
    store.record_auto_pass_review(profile_id="l1", would_like=True)
    store.undo_latest_auto_review()
    assert client.queries
    for sql, config in client.queries:
        bound = {p.name: p.value for p in (config.query_parameters if config else [])}
        for value in ("hinge", "l1", 7):
            assert f"'{value}'" not in sql and f'"{value}"' not in sql, sql
        for name in re.findall(r"@(\w+)", sql):
            assert name in bound, (name, sql)


def test_bq_a_migrated_store_never_probes_the_schema_on_a_read(bq):
    client, _storage, store = bq
    _bq_like(store, "l1")
    client.queries.clear()
    store.next_auto_pass_for_review(kind="likes")
    store.auto_review_waiting()
    store.auto_review_stats()
    store.record_auto_pass_review(profile_id="l1", would_like=True)
    assert not [sql for sql, _ in client.queries if "INFORMATION_SCHEMA" in sql]


def test_bq_a_read_only_store_probes_the_decision_column_once_per_process(monkeypatch):
    monkeypatch.setattr(bq_module, "_DECISION_COLUMN_TABLES", set())
    client, storage = _SqlBQ(), _Storage()
    runner = _bq_store(client, storage, ensure=True)
    _bq_like(runner, "l1")
    runner.close()
    bq_module._DECISION_COLUMN_TABLES.clear()  # a fresh process

    def probes():
        return [sql for sql, _ in client.queries if "INFORMATION_SCHEMA" in sql]

    client.queries.clear()
    hub = _bq_store(client, storage, ensure=False)
    assert hub.next_auto_pass_for_review(kind="likes")["profile_id"] == "l1"
    hub.auto_review_waiting()
    hub.close()
    second = _bq_store(client, storage, ensure=False)
    assert second.auto_review_stats()["likes"]["archived"] == 1
    second.close()
    assert len(probes()) == 1


def test_bq_missing_tables_read_as_empty_for_the_new_reads_of_a_read_only_store():
    client, storage = _SqlBQ(), _Storage()
    client.missing_tables = {"auto_passes", "auto_pass_photos", "auto_pass_reviews"}
    reader = _bq_store(client, storage, ensure=False)
    assert reader.next_auto_pass_for_review(kind="likes") is None
    assert reader.auto_review_waiting() == {
        "likes": 0,
        "random": 0,
        "near_cut": 0,
        "total": 0,
    }
    assert reader.auto_review_stats()["likes"] == {
        "archived": 0,
        "reviewed": 0,
        "agreed": 0,
    }


class _LegacySqlBQ(_SqlBQ):
    """A project whose `auto_passes` predates `decision`: no ensure=True store has started
    since the column was added, so only a read-only (hub) store has touched it."""

    def __init__(self):
        super().__init__()
        self.has_decision = False
        legacy = [c for c in _columns("auto_passes") if c != "decision"]
        self.con.execute('DROP TABLE "proj.ds.auto_passes"')
        self.con.execute(f'CREATE TABLE "proj.ds.auto_passes" ({", ".join(legacy)})')

    def query(self, sql, job_config=None):
        if "INFORMATION_SCHEMA.COLUMNS" in sql:
            self.queries.append((sql, job_config))
            return _Job(
                [
                    {"table_name": table, "column_name": col}
                    for table in _TABLES
                    for col in _columns(table)
                    if self.has_decision or (table, col) != ("auto_passes", "decision")
                ]
            )
        return super().query(sql, job_config)

    def migrate(self):
        self.con.execute('ALTER TABLE "proj.ds.auto_passes" ADD COLUMN decision')
        self.has_decision = True

    def archive(self, pid, *, random_sample, decision=None):
        row = {
            "run_id": "old-run",
            "app": "hinge",
            "profile_id": pid,
            "created_at": 1000.0,
            "score": 0.4,
            "threshold": 0.5,
            "embedding": [0.5],
            "photo_count": 1,
            "profile_name": "",
            "random_sample": random_sample,
            "capture_truncated": False,
        }
        if decision is not None:
            row["decision"] = decision
        self.insert("auto_passes", row)
        self.insert(
            "auto_pass_photos",
            {
                "run_id": "old-run",
                "app": "hinge",
                "profile_id": pid,
                "created_at": 1000.0,
                "photo_index": 0,
                "gcs_uri": f"gs://photos/{pid}.png",
                "sha256": "0" * 64,
                "byte_size": 1,
                "content_type": "image/png",
            },
        )


def test_bq_a_read_only_store_reads_an_unmigrated_archive_as_passes(monkeypatch):
    """The hub opens its store with ensure=False. Against a table no ensure=True store has
    migrated yet, a query naming `decision` fails ("Unrecognized name") and the review card
    would show an error; every row there predates archived likes, so they are passes."""
    monkeypatch.setattr(bq_module, "_DECISION_COLUMN_TABLES", set())
    client = _LegacySqlBQ()
    client.archive("r1", random_sample=True)
    client.archive("n1", random_sample=False)
    hub = _bq_store(client, _Storage(), ensure=False)

    assert hub.next_auto_pass_for_review(kind="likes") is None
    assert hub.next_auto_pass_for_review(kind="random")["profile_id"] == "r1"
    assert hub.auto_review_waiting() == {
        "likes": 0,
        "random": 1,
        "near_cut": 1,
        "total": 2,
    }
    stats = hub.auto_review_stats()
    assert (stats["pending_random"], stats["pending_near_cut"]) == (1, 1)
    assert stats["likes"]["archived"] == 0
    assert hub.record_auto_pass_review(profile_id="n1", would_like=True) == {
        "kind": "near_cut",
        "label_source": "auto_review_near_cut",
        "decision": "dislike",
        "score": 0.4,
        "threshold": 0.5,
        "would_like": True,
    }

    # Another process migrates the table and archives a like: the same store sees it (a
    # missing column is never remembered), and the like stays out of the pass pools.
    client.migrate()
    client.archive("l1", random_sample=False, decision="like")
    assert hub.next_auto_pass_for_review(kind="likes")["profile_id"] == "l1"
    assert hub.auto_review_waiting() == {
        "likes": 1,
        "random": 1,
        "near_cut": 0,
        "total": 2,
    }
    hub.close()


def test_bq_a_failing_schema_probe_raises_rather_than_guessing(monkeypatch):
    """Reading 'no decision column' after a failed probe would put archived likes into the
    near-cut pass pool, so the failure propagates."""
    monkeypatch.setattr(bq_module, "_DECISION_COLUMN_TABLES", set())

    class _ProbeFails(_SqlBQ):
        def query(self, sql, job_config=None):
            if "INFORMATION_SCHEMA.COLUMNS" in sql:
                raise RuntimeError("transient BigQuery failure")
            return super().query(sql, job_config)

    hub = _bq_store(_ProbeFails(), _Storage(), ensure=False)
    with pytest.raises(RuntimeError, match="transient"):
        hub.next_auto_pass_for_review(kind="near_cut")
    with pytest.raises(RuntimeError, match="transient"):
        hub.auto_review_waiting()


def test_bq_a_review_of_a_buffered_like_reads_its_decision_from_the_buffer(bq):
    client, _storage, store = bq  # flush_every=100: the manifest stays buffered
    _bq_like(store, "l1")
    assert client.count("auto_passes") == 0
    assert store._find_auto_pass("l1")["decision"] == "like"
    store.flush()
    assert json.loads(client.rows("auto_passes")[0]["embedding"]) == [0.1, 0.2]


def test_bq_like_recovery_returns_the_answer_actually_saved(bq):
    client, storage, store = bq
    _bq_like(store, "l1", score=0.92, threshold=0.63)
    store.flush()
    client.insert_failures["auto_pass_reviews"] = 99
    first = store.record_auto_pass_review(profile_id="l1", would_like=False)
    assert first["would_like"] is False
    assert client.count("labels") == 1 and client.count("auto_pass_reviews") == 0

    client.insert_failures["auto_pass_reviews"] = 0
    fresh = _bq_store(client, storage, ensure=False)
    recovered = fresh.record_auto_pass_review(profile_id="l1", would_like=True)

    assert recovered == {
        "kind": "likes",
        "label_source": "auto_review_like",
        "decision": "like",
        "score": 0.92,
        "threshold": 0.63,
        "would_like": False,
    }
    assert client.count("labels") == 1
    assert fresh.auto_review_stats()["likes"] == {
        "archived": 1,
        "reviewed": 1,
        "agreed": 0,
    }
