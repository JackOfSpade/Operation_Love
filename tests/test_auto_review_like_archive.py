"""AUTO review of LIKES, worker side: a landed model like is archived for the owner's review.

The owner reviews AUTO likes as well as passes, so the worker archives every model like that
landed.  The properties pinned here, each by a test that fails if its guard is removed:

* a like is archived only after ``driver.like()`` returned, its decision row exists and its
  opener was committed, with ``decision="like"``, ``random_sample=False`` and NO random-pool
  draw: likes must never enter the pass pools whose answers drive the live correction;
* review off archives nothing; demoted likes are never archived (tests/test_auto_review_worker);
* the archive is best-effort: a failure is loud and never stops the run or becomes a label;
* a store whose ``record_auto_pass`` has no ``decision`` keyword could only file a like as a
  pass, so it archives passes only and says so once per run;
* an archived like gets its ``photo_features`` row keyed by its lineage id;
* archived likes count towards the retention sweep.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from operation_love import worker as worker_mod
from operation_love.opener.opener import INDEX_SPACE_PROFILE_PHOTOS
from operation_love.opener.service import OpenerPick
from operation_love.ranker.decider import Decision
from operation_love.ranker.threshold import (
    FEATURE_SET_FACE_CROP_V1,
    FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1,
    fetch_label_rows,
    is_calibration_source,
)
from operation_love.vision.photo_cards import (
    WHOLE_PHOTO_DIMS,
    WHOLE_PHOTO_FEATURE_VERSION,
)
from operation_love.worker import Worker
from tests.test_auto_review_worker import (
    _Decider,
    _dislike,
    _Driver,
    _Pacing,
    _profile,
    _PruningStore,
    _run,
    _Store,
)


def _like(score=0.82, embedding=(0.4, 0.5, 0.6), **kwargs):
    return Decision("like", score, list(embedding), "ranker", **kwargs)


def _no_draw(share):
    raise AssertionError("a like must never draw a random-pool sample")


# --- what is archived ------------------------------------------------------------------------
def test_a_landed_model_like_is_archived_as_a_like_after_its_decision_row(monkeypatch):
    monkeypatch.setattr(worker_mod, "_draw_random_sample", _no_draw)
    profile = _profile(
        1, name="Blair", meta={"capture_truncated": True}, bio="not archived"
    )

    _, driver, store, events = _run([profile], _Decider([_like(0.82)]))

    # Like first, decision row second, archive last: the archive is conditional on both.
    assert events == ["like", "decision", "auto_pass"]
    assert store.auto_passes == [
        {
            "run_id": "run-1",
            "app": "bumble",
            # The same opaque lineage id the decision row carries.
            "profile_id": store.decision_kwargs[0]["profile_id"],
            "score": 0.82,
            "threshold": 0.61,
            "embedding": [0.4, 0.5, 0.6],  # the v1 vector, as for a pass
            "photos": [b"png-1-a", b"png-1-b"],
            "profile_name": "Blair",
            "capture_truncated": True,
            "decision": "like",
            "random_sample": False,  # never in the random PASS pool
        }
    ]
    # An unreviewed like says nothing about the owner's taste: no label, no profile row.
    assert store.labels == [] and store.profiles == []


def test_likes_and_passes_each_get_their_own_lineage_and_only_passes_draw(
    monkeypatch,
):
    draws = []

    def _draw(share):
        draws.append(share)
        return len(draws) == 1

    monkeypatch.setattr(worker_mod, "_draw_random_sample", _draw)

    _, _, store, _ = _run(
        [_profile(n) for n in range(4)],
        _Decider([_like(0.9), _dislike(0.2), _like(0.7), _dislike(0.3)]),
    )

    assert draws == [0.5, 0.5]  # one draw per PASS, none per like
    assert [
        (p.get("decision", "dislike"), p["random_sample"]) for p in store.auto_passes
    ] == [("like", False), ("dislike", True), ("like", False), ("dislike", False)]
    # Passes keep the long-standing call shape: no decision keyword (the store default).
    assert ["decision" in p for p in store.auto_passes] == [True, False, True, False]
    ids = [p["profile_id"] for p in store.auto_passes]
    assert len(set(ids)) == 4
    assert ids == [kw["profile_id"] for kw in store.decision_kwargs]


def test_nothing_is_archived_for_a_like_when_review_is_off():
    _, _, store, events = _run([_profile()], _Decider([_like()]), share=None)

    assert events == ["like", "decision"]
    assert store.auto_passes == []


def test_a_like_without_an_embedding_is_not_archived():
    _, _, store, events = _run(
        [_profile()], _Decider([Decision("like", 0.9, [], "ranker")])
    )

    assert events == ["like", "decision"]
    assert store.auto_passes == []


def test_the_like_archive_is_never_attempted_when_the_decision_row_cannot_be_written():
    events = []
    store = _Store(events)

    def _fail(*a, **k):
        raise RuntimeError("store down")

    store.record_decision = _fail

    _run([_profile()], _Decider([_like()]), store=store, events=events)

    assert events == ["like"]  # the like landed; its row did not; so no archive
    assert store.auto_passes == []


# --- ordering against the opener commit --------------------------------------------------------
class _OpenerDriver(_Driver):
    accepts_opener = True

    def like(self, opener=None, item_index=None, *, model_item_index=None):
        self.events.append(("like", opener))


class _StagingOpenerService:
    """The staged opener protocol: maybe_opener(stage=True) then commit_opener(lineage)."""

    disabled = False
    stop_requested = False

    def __init__(self, events):
        self.events = events
        self.commits = []

    def maybe_opener(
        self, run_id, app, profile, *, items=None, should_stop=None, stage=False
    ):
        self.events.append("opener")
        return OpenerPick("hi there", index=1, index_space=INDEX_SPACE_PROFILE_PHOTOS)

    def commit_opener(self, pick, **lineage):
        self.events.append("commit")
        self.commits.append(lineage)


def test_a_like_is_archived_only_after_its_opener_was_committed():
    events: list = []
    store = _Store(events)
    service = _StagingOpenerService(events)

    Worker(
        "bumble",
        _OpenerDriver([_profile(1)], events),
        _Decider([_like()]),
        service,
        store,
        "run-1",
        _Pacing(),
        threading.Event(),
        mode="auto",
        auto_review_random_share=0.5,
    ).run()

    # The committed opener is the durable record of what was sent; the review archive is an
    # add-on that comes after it and can never delay or replace it.
    assert events == [
        "opener",
        ("like", "hi there"),
        "decision",
        "commit",
        "auto_pass",
    ]
    assert service.commits[0]["profile_id"] == store.auto_passes[0]["profile_id"]
    assert store.auto_passes[0]["decision"] == "like"


@pytest.mark.parametrize("where", ["like", "commit"])
def test_a_failed_like_or_opener_commit_never_archives_the_like(where):
    events: list = []
    store = _Store(events)
    service = _StagingOpenerService(events)
    driver = _OpenerDriver([_profile(1)], events)

    def fail(*args, **kwargs):
        events.append(f"{where}-failed")
        raise RuntimeError(
            "like did not complete" if where == "like" else "commit failed"
        )

    if where == "like":
        driver.like = fail
    else:
        service.commit_opener = fail
    Worker(
        "bumble",
        driver,
        _Decider([_like()]),
        service,
        store,
        "run-1",
        _Pacing(),
        threading.Event(),
        mode="auto",
        auto_review_random_share=0.5,
    ).run()

    assert f"{where}-failed" in events
    assert store.auto_passes == []
    assert len(store.decisions) == (0 if where == "like" else 1)
    assert store.labels == []


# --- failures --------------------------------------------------------------------------------
def test_like_archive_failures_are_loud_and_never_stop_the_run_or_become_labels(
    capsys,
):
    outcomes = iter([RuntimeError("bucket unreachable"), False, None, True])

    def _auto_pass(**kw):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    events = []
    store = _Store(events, auto_pass=_auto_pass)

    _, driver, store, events = _run(
        [_profile(n) for n in range(4)],
        _Decider([_like()]),
        store=store,
        events=events,
    )

    out = capsys.readouterr().out
    # Every profile was still liked and audited; the run ended because the deck did.
    assert events.count("like") == 4 and len(store.decisions) == 4
    assert driver.out_of_profiles()
    assert store.labels == [] and store.profiles == []
    assert "RuntimeError: bucket unreachable" in out
    assert "returned False" in out and "returned None" in out
    assert out.count("auto like review:") == 3  # the 4th archived cleanly
    assert "auto pass review" not in out


def test_a_like_with_no_photos_is_reported_not_archived(capsys):
    _, _, store, events = _run([_profile(photos=[])], _Decider([_like()]))

    assert events == ["like", "decision"]
    assert store.auto_passes == []
    assert "this like captured no photos" in capsys.readouterr().out


class _PassOnlyStore(_Store):
    """A store that predates like review: an explicit signature with no ``decision``."""

    def record_auto_pass(
        self,
        *,
        run_id,
        app,
        profile_id,
        score,
        threshold,
        embedding,
        photos,
        profile_name="",
        random_sample,
        capture_truncated=False,
    ):
        self.events.append("auto_pass")
        self.auto_passes.append({"profile_id": profile_id, "score": score})
        return True


def test_a_store_that_cannot_file_a_like_archives_passes_only_and_says_so_once(capsys):
    events: list = []
    store = _PassOnlyStore(events)

    _, _, store, events = _run(
        [_profile(n) for n in range(4)],
        _Decider([_like(0.9), _dislike(0.2), _like(0.8), _dislike(0.3)]),
        store=store,
        events=events,
    )

    # Filing a like on such a store would put it in a pass pool; it is skipped instead.
    assert [p["score"] for p in store.auto_passes] == [0.2, 0.3]
    assert events.count("like") == 2 and events.count("dislike") == 2
    assert capsys.readouterr().out.count("takes no decision keyword") == 1


def test_a_store_without_record_auto_pass_warns_once_for_likes_and_passes(capsys):
    from tests.test_auto_review_worker import _StoreWithoutArchive

    events: list = []
    store = _StoreWithoutArchive(events)

    _run(
        [_profile(n) for n in range(3)],
        _Decider([_like(0.9), _dislike(0.2), _like(0.8)]),
        store=store,
        events=events,
    )

    out = capsys.readouterr().out
    assert store.decisions == ["like", "dislike", "like"]
    assert out.count("has no record_auto_pass") == 1
    assert "passes and likes will NOT be archived" in out


# --- whole-photo feature rows ------------------------------------------------------------------
class _PhotoDecider(_Decider):
    def __init__(self, decisions, *, feature_set=FEATURE_SET_FACE_CROP_V1, error=None):
        super().__init__(decisions)
        self.feature_set = feature_set
        self.error = error
        self.whole_profiles = []

    def embed_whole_photos(self, profile):
        self.whole_profiles.append(profile)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            vector=[0.75] * WHOLE_PHOTO_DIMS,
            card_count=5,
            frames_sha256="cd" * 32,
            feature_version=WHOLE_PHOTO_FEATURE_VERSION,
        )


def _photo_store(events, **kwargs):
    store = _Store(events, **kwargs)
    rows = []
    store.add_photo_features = lambda **row: rows.append(row) or True
    return store, rows


def test_an_archived_v1_like_gets_its_photo_row_keyed_by_its_lineage_id():
    events: list = []
    store, rows = _photo_store(events)
    profile = _profile(1)
    decider = _PhotoDecider([_like(0.8, embedding=(0.1, 0.2))])

    _run([profile], decider, store=store, events=events)

    assert store.auto_passes[0]["embedding"] == [0.1, 0.2]  # v1 stays in auto_passes
    assert decider.whole_profiles == [profile]
    assert rows == [
        {
            "run_id": "run-1",
            "app": "bumble",
            "profile_id": store.auto_passes[0]["profile_id"],
            "feature_version": WHOLE_PHOTO_FEATURE_VERSION,
            "embedding": [0.75] * WHOLE_PHOTO_DIMS,
            "card_count": 5,
            "frames_sha256": "cd" * 32,
            "source": "live",
            "best_effort": True,
        }
    ]
    assert store.labels == []


def test_an_archived_whole_photo_like_reuses_the_decisions_vector():
    events: list = []
    store, rows = _photo_store(events)
    decision = _like(
        0.8,
        embedding=(0.1, 0.2),
        photo_embedding=[0.5] * WHOLE_PHOTO_DIMS,
        photo_card_count=4,
        frames_sha256="ef" * 32,
    )
    decider = _PhotoDecider(
        [decision], feature_set=FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1
    )

    _run([_profile(1)], decider, store=store, events=events)

    assert decider.whole_profiles == []  # computed once, in decide()
    assert store.auto_passes[0]["embedding"] == [0.1, 0.2]
    assert rows[0]["embedding"] == [0.5] * WHOLE_PHOTO_DIMS
    assert rows[0]["card_count"] == 4 and rows[0]["frames_sha256"] == "ef" * 32
    assert rows[0]["profile_id"] == store.auto_passes[0]["profile_id"]
    assert rows[0]["best_effort"] is True


@pytest.mark.parametrize("where", ["embed", "store"])
def test_a_failed_like_photo_row_never_stops_the_run(where, capsys):
    events: list = []
    store = _Store(events)

    def broken(**row):
        raise RuntimeError("store broke")

    store.add_photo_features = (
        broken if where == "store" else (lambda **row: pytest.fail("not computed"))
    )
    decider = _PhotoDecider(
        [_like(0.9), _like(0.8)],
        error=RuntimeError("clip broke") if where == "embed" else None,
    )

    _run([_profile(1), _profile(2)], decider, store=store, events=events)

    assert len(store.auto_passes) == 2 and events.count("like") == 2
    out = capsys.readouterr().out
    assert out.count("WARNING") == 2
    assert out.count("auto like review") == 2
    assert ("clip broke" if where == "embed" else "store broke") in out


def test_a_rejected_like_archive_writes_no_photo_row():
    events: list = []
    store, rows = _photo_store(events, auto_pass=lambda **kw: False)

    _run([_profile(1)], _PhotoDecider([_like()]), store=store, events=events)

    assert rows == []


# --- retention -------------------------------------------------------------------------------
def test_archived_likes_count_towards_the_retention_sweep(monkeypatch):
    monkeypatch.setattr(worker_mod, "_AUTO_PRUNE_EVERY", 2)
    events: list = []
    store = _PruningStore(events)

    _run(
        [_profile(n) for n in range(4)],
        _Decider([_like(0.9), _like(0.8), _dislike(0.2), _like(0.7)]),
        store=store,
        events=events,
        auto_review_max_passes=5,
    )

    assert len(store.auto_passes) == 4
    # Start, after the 2nd archive (two likes) and after the 4th.
    assert events.count("prune") == 3
    positions = [i for i, e in enumerate(events) if e == "prune"]
    assert [events[i - 1] for i in positions[1:]] == ["auto_pass", "auto_pass"]


# --- the real store ----------------------------------------------------------------------------
def test_the_worker_archives_likes_into_the_real_sqlite_store_outside_the_pass_pools(
    tmp_path,
):
    """Catches drift between the keywords the worker sends and what the store accepts, and
    pins that an archived like is reviewable in the likes pool and in no pass pool."""
    from operation_love.ranker.store import SQLiteStore

    store = SQLiteStore(tmp_path / "love.db")
    try:
        Worker(
            "bumble",
            _Driver([_profile(n, name=f"P{n}") for n in range(2)], []),
            _Decider([_like(0.8), _dislike(0.2)], threshold=0.55),
            None,
            store,
            "run-real",
            _Pacing(),
            threading.Event(),
            mode="auto",
            auto_review_random_share=1.0,
        ).run()

        decision_ids = [
            row[0]
            for row in store.con.execute(
                "SELECT profile_id FROM decisions ORDER BY rowid"
            ).fetchall()
        ]
        like_id, pass_id = decision_ids
        assert store.auto_review_waiting() == {
            "likes": 1,
            "random": 1,
            "near_cut": 0,
            "total": 2,
        }
        liked = store.next_auto_pass_for_review(kind="likes")
        assert liked["profile_id"] == like_id and liked["photo_count"] == 2
        assert store.next_auto_pass_for_review(kind="random")["profile_id"] == pass_id
        assert store.next_auto_pass_for_review(kind="near_cut") is None
        stats = store.auto_review_stats()
        # The like is in no pass count: the live correction reads passes only.
        assert stats["pending_random"] == 1 and stats["pending_near_cut"] == 0
        assert store.load_labels() == []  # an unanswered like is not training data
    finally:
        store.close()


@pytest.mark.parametrize(
    "feature_set", [FEATURE_SET_FACE_CROP_V1, FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1]
)
def test_a_reviewed_auto_like_keeps_v1_and_whole_photo_vectors_separate(
    tmp_path, feature_set
):
    from operation_love.ranker.store import SQLiteStore

    v1 = [0.2] * 1280
    whole = [0.75] * WHOLE_PHOTO_DIMS
    extras = (
        {
            "photo_embedding": whole,
            "photo_card_count": 5,
            "frames_sha256": "cd" * 32,
        }
        if feature_set == FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1
        else {}
    )
    decider = _PhotoDecider([_like(embedding=v1, **extras)], feature_set=feature_set)
    store = SQLiteStore(tmp_path / "love.db")
    profile = _profile(1)
    try:
        _run([profile], decider, store=store)

        archived = store.next_auto_pass_for_review(kind="likes")
        assert archived is not None
        assert fetch_label_rows(store, feature_set) == []
        assert decider.whole_profiles == (
            [profile] if feature_set == FEATURE_SET_FACE_CROP_V1 else []
        )
        store.record_auto_pass_review(
            profile_id=archived["profile_id"], would_like=False
        )

        assert store.load_label_rows() == [(False, v1, "auto_review_like")]
        assert fetch_label_rows(store, FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1) == [
            (False, v1 + whole, "auto_review_like")
        ]
        assert is_calibration_source("auto_review_like") is False
    finally:
        store.close()
