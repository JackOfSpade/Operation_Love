"""AUTO pass review, worker side: which passes are archived, when, and what a failure may cost.

The archive exists so the owner can later say "I would have liked that one".  Three properties
matter and each has a test that fails if the guard behind it is removed:

* only a REAL model dislike is archived (not a like, not a policy-demoted like, not no_face),
  and only AFTER the pass landed and its decision row exists;
* the archive can never stop the run, and never turns into a training label;
* the contextual-demotion headroom the model grants reaches AutoSessionPolicy.
"""

import math
import threading

import pytest

from operation_love import worker as worker_mod
from operation_love.drivers.base import DatingAppDriver
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import Decision
from operation_love.worker import Worker


class _Pacing:
    swipe_delay_s = 0.0


class _Driver(DatingAppDriver):
    """Hands out a fixed list of profiles and logs every physical action in `events`."""

    accepts_opener = False
    supports_interruptible_dislike = True

    def __init__(self, profiles, events):
        self.profiles = list(profiles)
        self.events = events
        self.i = 0

    def open_session(self):
        pass

    def next_profile(self):
        if self.i >= len(self.profiles):
            return None
        profile = self.profiles[self.i]
        self.i += 1
        return profile

    def like(self, opener=None, item_index=None, *, model_item_index=None):
        self.events.append("like")

    def dislike(self, *, should_stop=None):
        self.events.append("dislike")

    def out_of_profiles(self):
        return self.i >= len(self.profiles)

    def close(self):
        pass


class _Store:
    """Just enough store for an AUTO dislike/like run, plus a controllable record_auto_pass."""

    def __init__(self, events, *, auto_pass=None):
        self.events = events
        self.decisions = []
        self.decision_kwargs = []
        self.labels = []
        self.profiles = []
        self.auto_passes = []
        self._auto_pass = auto_pass  # callable(**kw) -> bool | raises

    def load_labels(self):
        return []

    def count_today(self, app):
        return 0

    def record_decision(self, run_id, app, decision, score, source="auto", **kw):
        self.events.append("decision")
        self.decisions.append((app, decision, source))
        self.decision_kwargs.append(kw)

    def add_label(self, *a, **k):
        self.labels.append((a, k))

    def record_profile(self, *a, **k):
        self.profiles.append((a, k))
        return True

    def record_auto_pass(self, **kw):
        self.events.append("auto_pass")
        self.auto_passes.append(kw)
        if self._auto_pass is not None:
            return self._auto_pass(**kw)
        return True

    def flush(self):
        pass

    def close(self):
        pass


class _StoreWithoutArchive:
    """A store that predates the review archive: no record_auto_pass at all."""

    def __init__(self, events):
        self.events = events
        self.decisions = []

    def load_labels(self):
        return []

    def count_today(self, app):
        return 0

    def record_decision(self, run_id, app, decision, score, source="auto", **kw):
        self.decisions.append(decision)

    def flush(self):
        pass

    def close(self):
        pass


class _Decider:
    """Returns the scripted decisions in order; carries a model with a threshold."""

    def __init__(self, decisions, *, threshold=0.61, lift=None):
        self.decisions = list(decisions)
        self.calls = 0

        class _Model:
            pass

        self.model = _Model()
        self.model.threshold = threshold
        if lift is not None:
            self.model.max_threshold_lift = lift

    def decide(self, profile):
        decision = self.decisions[min(self.calls, len(self.decisions) - 1)]
        self.calls += 1
        return decision


def _dislike(score=0.30, embedding=(0.1, 0.2, 0.3)):
    return Decision("dislike", score, list(embedding), "ranker")


def _profile(n=0, **kwargs):
    kwargs.setdefault("photos", [f"png-{n}-a".encode(), f"png-{n}-b".encode()])
    return Profile(**kwargs)


def _run(profiles, decider, *, share=0.5, store=None, events=None, **worker_kwargs):
    events = events if events is not None else []
    driver = _Driver(profiles, events)
    store = store if store is not None else _Store(events)
    worker = Worker(
        "bumble",
        driver,
        decider,
        None,
        store,
        "run-1",
        _Pacing(),
        threading.Event(),
        mode="auto",
        auto_review_random_share=share,
        **worker_kwargs,
    )
    worker.run()
    return worker, driver, store, events


# --- what is archived ------------------------------------------------------------------------
def test_model_dislike_is_archived_after_the_pass_landed_and_its_decision_row_exists(
    monkeypatch,
):
    monkeypatch.setattr(worker_mod, "_draw_random_sample", lambda share: True)
    profile = _profile(
        1, name="Avery", meta={"capture_truncated": True}, bio="not archived"
    )

    _, driver, store, events = _run([profile], _Decider([_dislike(0.30)]))

    # Pass first, decision row second, archive last: the archive is conditional on both.
    assert events == ["dislike", "decision", "auto_pass"]
    assert len(store.auto_passes) == 1
    archived = store.auto_passes[0]
    assert archived == {
        "run_id": "run-1",
        "app": "bumble",
        # The same opaque lineage id the decision row carries, so a reviewed pass joins back
        # to the action that produced it.
        "profile_id": store.decision_kwargs[0]["profile_id"],
        "score": 0.30,
        "threshold": 0.61,
        "embedding": [0.1, 0.2, 0.3],
        "photos": [b"png-1-a", b"png-1-b"],
        "profile_name": "Avery",
        "random_sample": True,
        "capture_truncated": True,
    }
    assert len(archived["profile_id"]) == 32
    assert (
        int(archived["profile_id"], 16) >= 0
    )  # uuid4().hex: what the hub's validator accepts
    # An unreviewed pass says nothing about the owner's taste: no label, no profile row.
    assert store.labels == [] and store.profiles == []


def test_the_photos_archived_are_the_profile_photos_exactly():
    photos = [b"\x89PNG-one", b"\x89PNG-two", b"\x89PNG-three"]

    _, _, store, _ = _run([_profile(photos=photos)], _Decider([_dislike()]))

    assert store.auto_passes[0]["photos"] == photos


def test_each_pass_gets_its_own_lineage_id_and_a_fresh_random_draw(monkeypatch):
    draws = iter([True, False, True])
    monkeypatch.setattr(worker_mod, "_draw_random_sample", lambda share: next(draws))

    _, _, store, _ = _run(
        [_profile(n) for n in range(3)], _Decider([_dislike(0.2), _dislike(0.4)])
    )

    assert [p["random_sample"] for p in store.auto_passes] == [True, False, True]
    ids = [p["profile_id"] for p in store.auto_passes]
    assert len(set(ids)) == 3
    assert ids == [kw["profile_id"] for kw in store.decision_kwargs]


def test_draw_random_sample_is_a_bernoulli_draw_at_the_configured_share(monkeypatch):
    class _Rng:
        value = 0.0

        @classmethod
        def random(cls):
            return cls.value

    monkeypatch.setattr(worker_mod, "random", _Rng)
    _Rng.value = 0.4999
    assert worker_mod._draw_random_sample(0.5) is True
    _Rng.value = 0.5
    assert worker_mod._draw_random_sample(0.5) is False
    # The closed ends are honest probabilities: never / always, whatever the draw is.
    for value in (0.0, 0.7, 0.999999):
        _Rng.value = value
        assert worker_mod._draw_random_sample(0.0) is False
        assert worker_mod._draw_random_sample(1.0) is True


def test_a_like_is_never_archived():
    like = Decision("like", 0.95, [0.1, 0.2], "ranker")

    _, driver, store, events = _run([_profile()], _Decider([like]))

    assert "like" in events and store.auto_passes == []


def test_no_face_is_never_archived_even_though_it_lands_as_a_pass():
    no_face = Decision("no_face", 0.0, [], "ranker")

    _, driver, store, events = _run([_profile()], _Decider([no_face]))

    # The conservative pass still happens and is still audited as no_face ...
    assert "dislike" in events
    assert store.decisions == [("bumble", "no_face", "auto")]
    # ... but there is no embedding to learn from and nothing to review.
    assert store.auto_passes == []


def test_a_dislike_without_an_embedding_is_not_archived():
    _, _, store, events = _run(
        [_profile()], _Decider([Decision("dislike", 0.2, [], "ranker")])
    )

    assert events == ["dislike", "decision"]
    assert store.auto_passes == []


def test_a_policy_demoted_like_is_not_archived(monkeypatch):
    """A demoted like scored AT OR ABOVE the cut; archiving it would inflate the measured
    false-pass rate the live recall adjustment is calibrated against."""

    class _DemotingPolicy:
        def __init__(self, *, base_threshold):
            pass

        def apply_decision(self, decision, profile):
            class _Result:
                pass

            result = _Result()
            result.decision = Decision(
                "dislike", decision.score, decision.embedding, "ranker_contextual"
            )
            return result

        def record_landed_action(self, decision):
            pass

    monkeypatch.setattr(worker_mod, "AutoSessionPolicy", _DemotingPolicy)
    like = Decision("like", 0.70, [0.1, 0.2], "ranker")

    _, driver, store, events = _run([_profile()], _Decider([like]))

    assert events == ["dislike", "decision"]  # it WAS passed, and audited as a pass ...
    assert store.decisions == [("bumble", "dislike", "auto")]
    assert store.auto_passes == []  # ... but it is not a model dislike


def test_a_like_demoted_by_the_ratio_ceiling_is_not_archived_either():
    """The other demotion path: marginal likes are turned into passes to keep the right-swipe
    ratio human.  They scored above the cut, so they are not model dislikes either."""
    from operation_love.limits import RateLimiter

    marginal = Decision(
        "like", 0.55, [0.1, 0.2], "ranker"
    )  # clears the policy's 0.545 ceiling
    genuine_pass = _dislike(0.30)
    events = []
    driver = _Driver([_profile(n) for n in range(3)], events)
    store = _Store(events)

    Worker(
        "bumble",
        driver,
        _Decider([marginal, marginal, genuine_pass], threshold=0.5),
        None,
        store,
        "run-1",
        _Pacing(),
        threading.Event(),
        mode="auto",
        limiter=RateLimiter(target_like_ratio=0.5),
        auto_review_random_share=0.5,
    ).run()

    # card 1 liked; card 2 demoted to a pass by the ceiling; card 3 a real model dislike.
    assert [d for _, d, _ in store.decisions] == ["like", "dislike", "dislike"]
    assert [p["score"] for p in store.auto_passes] == [0.30]


def test_nothing_is_archived_when_review_is_off():
    _, _, store, events = _run([_profile()], _Decider([_dislike()]), share=None)

    assert events == ["dislike", "decision"]
    assert store.auto_passes == []


def test_the_archive_is_never_attempted_when_the_decision_row_cannot_be_written():
    events = []
    store = _Store(events)

    def _fail(*a, **k):
        raise RuntimeError("store down")

    store.record_decision = _fail

    worker, driver, store, events = _run(
        [_profile()], _Decider([_dislike()]), store=store, events=events
    )

    assert events == ["dislike"]  # the pass landed; its row did not; so no archive
    assert store.auto_passes == []


# --- failures --------------------------------------------------------------------------------
def test_archive_failures_are_loud_and_never_stop_the_run_or_become_labels(capsys):
    outcomes = iter([RuntimeError("bucket unreachable"), False, None, True])

    def _auto_pass(**kw):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    events = []
    store = _Store(events, auto_pass=_auto_pass)

    worker, driver, store, events = _run(
        [_profile(n) for n in range(4)],
        _Decider([_dislike()]),
        store=store,
        events=events,
    )

    out = capsys.readouterr().out
    # Every profile was still passed and audited; the run ended because the deck did.
    assert events.count("dislike") == 4 and len(store.decisions) == 4
    assert driver.out_of_profiles()
    # None of the archive failures became a label.
    assert store.labels == [] and store.profiles == []
    # Loud: the exception, the explicit False and a non-True return are each reported.
    assert "RuntimeError: bucket unreachable" in out
    assert "returned False" in out
    assert "returned None" in out
    assert out.count("auto pass review:") == 3  # the 4th archived cleanly


def test_a_store_without_record_auto_pass_warns_once_and_the_run_continues(capsys):
    events = []
    store = _StoreWithoutArchive(events)

    _, driver, _, events = _run(
        [_profile(n) for n in range(3)],
        _Decider([_dislike()]),
        store=store,
        events=events,
    )

    out = capsys.readouterr().out
    assert events.count("dislike") == 3 and len(store.decisions) == 3
    assert out.count("has no record_auto_pass") == 1


def test_a_pass_with_no_photos_is_reported_not_archived(capsys):
    _, _, store, events = _run([_profile(photos=[])], _Decider([_dislike()]))

    assert store.auto_passes == []
    assert "captured no photos" in capsys.readouterr().out


# --- construction ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad", [True, False, "0.5", -0.01, 1.01, float("nan"), float("inf"), [0.5]]
)
def test_an_ill_typed_or_out_of_range_share_is_refused_at_construction(bad):
    with pytest.raises(ValueError, match="auto_review_random_share"):
        Worker(
            "bumble",
            _Driver([], []),
            _Decider([_dislike()]),
            None,
            _Store([]),
            "run-1",
            _Pacing(),
            threading.Event(),
            mode="auto",
            auto_review_random_share=bad,
        )


@pytest.mark.parametrize("good", [None, 0, 0.0, 0.5, 1, 1.0])
def test_a_probability_or_none_is_accepted(good):
    worker = Worker(
        "bumble",
        _Driver([], []),
        _Decider([_dislike()]),
        None,
        _Store([]),
        "run-1",
        _Pacing(),
        threading.Event(),
        mode="auto",
        auto_review_random_share=good,
    )
    assert worker.auto_review_random_share == (None if good is None else float(good))


# --- the model's contextual-demotion headroom ------------------------------------------------
def _policy_probe(monkeypatch):
    made = []

    class _Probe:
        def __init__(self, **kwargs):
            made.append(kwargs)

        def apply_decision(self, decision, profile):
            class _R:
                pass

            r = _R()
            r.decision = decision
            return r

        def record_landed_action(self, decision):
            pass

    monkeypatch.setattr(worker_mod, "AutoSessionPolicy", _Probe)
    return made


def test_the_model_headroom_reaches_the_session_policy(monkeypatch):
    made = _policy_probe(monkeypatch)

    _run([_profile()], _Decider([_dislike()], threshold=0.47, lift=0.012))

    assert made == [{"base_threshold": 0.47, "max_threshold_lift": 0.012}]


def test_a_model_without_the_property_leaves_the_policy_untouched(monkeypatch):
    made = _policy_probe(monkeypatch)

    _run([_profile()], _Decider([_dislike()], threshold=0.47))

    # No keyword at all: a legacy model / fake decider constructs the policy exactly as before.
    assert made == [{"base_threshold": 0.47}]


def test_a_zero_headroom_disables_the_real_policys_demotion():
    """End to end with the REAL AutoSessionPolicy: lift 0.0 means a like is never demoted."""
    like = Decision("like", 0.5101, [0.1, 0.2], "ranker")

    _, driver, store, events = _run(
        [_profile(n) for n in range(25)],
        _Decider([like], threshold=0.51, lift=0.0),
    )

    assert events.count("like") == 25
    assert store.auto_passes == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, "x", True, [0.1]])
def test_a_present_but_unusable_headroom_means_no_demotion_and_says_so(bad, capsys):
    model = type("M", (), {"max_threshold_lift": bad})()

    assert worker_mod._session_max_threshold_lift(model) == 0.0
    assert "max_threshold_lift" in capsys.readouterr().out


def test_headroom_is_capped_at_the_policys_own_ceiling():
    from operation_love.interaction import _THRESHOLD_MAX_LIFT

    model = type("M", (), {"max_threshold_lift": 0.9})()
    assert worker_mod._session_max_threshold_lift(model) == _THRESHOLD_MAX_LIFT
    assert math.isclose(
        worker_mod._session_max_threshold_lift(
            type("M", (), {"max_threshold_lift": 0.02})()
        ),
        0.02,
    )
    # Absent -> the historical cap, bit for bit.
    assert worker_mod._session_max_threshold_lift(object()) == _THRESHOLD_MAX_LIFT


# --- the real store ----------------------------------------------------------------------------
def test_the_worker_archives_into_the_real_sqlite_store_without_writing_labels(
    tmp_path,
):
    """Catches any drift between the keywords the worker sends and what the store accepts."""
    from operation_love.ranker.store import SQLiteStore

    store = SQLiteStore(tmp_path / "love.db")
    try:
        events = []
        driver = _Driver([_profile(n, name=f"P{n}") for n in range(3)], events)
        worker = Worker(
            "bumble",
            driver,
            _Decider([_dislike(0.2), _dislike(0.4), _dislike(0.3)], threshold=0.55),
            None,
            store,
            "run-real",
            _Pacing(),
            threading.Event(),
            mode="auto",
            auto_review_random_share=1.0,
        )
        worker.run()

        stats = store.auto_review_stats()
        assert stats["pending_random"] == 3 and stats["pending_near_cut"] == 0
        # The decisions the archive is measured against were recorded too.  The worker records
        # a pass's decision row BEFORE its archive row, so the stats window must open at the
        # first archived pass's own decision: all three passes count (it used to read 2).
        assert (
            store.con.execute(
                "SELECT COUNT(*) FROM decisions WHERE source='auto' AND decision='dislike'"
            ).fetchone()[0]
            == 3
        )
        assert stats["auto_passes"] == 3
        assert stats["auto_likes"] == 0
        assert store.load_labels() == []  # an unanswered pass is not training data

        item = store.next_auto_pass_for_review(kind="random")
        assert item["kind"] == "random" and item["photo_count"] == 2
        assert item["threshold"] == pytest.approx(0.55)
        assert store.auto_pass_photo(profile_id=item["profile_id"], index=1) is not None
        # The pass joins back to the decision row that recorded it.
        decision_ids = {
            row[0]
            for row in store.con.execute("SELECT profile_id FROM decisions").fetchall()
        }
        assert item["profile_id"] in decision_ids
    finally:
        store.close()


def test_a_share_of_zero_archives_only_near_cut_passes_in_the_real_store(tmp_path):
    from operation_love.ranker.store import SQLiteStore

    store = SQLiteStore(tmp_path / "love.db")
    try:
        worker = Worker(
            "bumble",
            _Driver([_profile(n) for n in range(2)], []),
            _Decider([_dislike(0.2)], threshold=0.55),
            None,
            store,
            "run-real",
            _Pacing(),
            threading.Event(),
            mode="auto",
            auto_review_random_share=0.0,
        )
        worker.run()

        stats = store.auto_review_stats()
        assert stats["pending_random"] == 0 and stats["pending_near_cut"] == 2
    finally:
        store.close()


# --- a failed calibration mid-run must be visible ----------------------------------------------
class _RetrainDecider:
    def __init__(self, ready, reason=None):
        self._ready = ready

        class _Model:
            pass

        self.model = _Model()
        if reason is not None:
            self.model.not_ready_reason = reason

    def retrain(self, store):
        return self._ready


def _retrain_worker(decider):
    return Worker(
        "bumble",
        _Driver([], []),
        decider,
        None,
        _Store([]),
        "run-1",
        _Pacing(),
        threading.Event(),
        mode="auto",
    )


def test_a_model_that_cannot_calibrate_says_why_after_a_retrain(capsys):
    reason = "recall target 0.95 needs at least 2 identity groups with both classes"
    _retrain_worker(_RetrainDecider(False, reason))._retrain_after_labels(3)

    out = capsys.readouterr().out
    assert "ranker ready=False" in out
    assert f"Ranker not ready: {reason}" in out


def test_a_ready_model_or_a_double_without_a_reason_prints_no_reason_line(capsys):
    _retrain_worker(_RetrainDecider(True, "stale reason"))._retrain_after_labels(1)
    _retrain_worker(_RetrainDecider(False))._retrain_after_labels(
        1
    )  # no attribute at all

    out = capsys.readouterr().out
    assert "ranker ready=True" in out and "ranker ready=False" in out
    assert "Ranker not ready" not in out


# --- retention -------------------------------------------------------------------------------
class _PruningStore(_Store):
    """A store that also bounds its archive: logs every ``prune_auto_passes`` call."""

    def __init__(self, events, *, prune=None, **kwargs):
        super().__init__(events, **kwargs)
        self.prunes = []
        self._prune = prune

    def prune_auto_passes(self, **kw):
        self.events.append("prune")
        self.prunes.append(kw)
        if self._prune is not None:
            return self._prune(**kw)
        return 0


def test_retention_sweeps_when_the_run_starts_and_every_few_archived_passes(
    monkeypatch,
):
    """The archive holds strangers' screenshots, so it is bounded from the first pass.  A sweep
    runs BEFORE the first profile (the backlog of earlier runs) and then every N archived passes;
    removing the call or the counter fails the count and the position of the first sweep."""
    monkeypatch.setattr(worker_mod, "_AUTO_PRUNE_EVERY", 3)
    events: list = []
    store = _PruningStore(events)
    profiles = [_profile(i) for i in range(7)]

    _, _, store, events = _run(
        profiles,
        _Decider([_dislike()]),
        store=store,
        events=events,
        auto_review_max_passes=5,
        auto_review_max_age_days=10,
    )

    assert len(store.auto_passes) == 7
    assert events[0] == "prune"  # before any profile was touched
    assert events.count("prune") == 3  # start, after the 3rd and after the 6th archive
    assert store.prunes == [{"max_passes": 5, "max_age_days": 10.0}] * 3
    # The mid-run sweeps sit right after an archive, never between a pass and its archive.
    positions = [i for i, e in enumerate(events) if e == "prune"]
    assert [events[i - 1] for i in positions[1:]] == ["auto_pass", "auto_pass"]
    assert events[: positions[1]].count("auto_pass") == 3


def test_unbounded_retention_never_touches_the_store():
    events: list = []
    store = _PruningStore(events)
    _run([_profile(1)], _Decider([_dislike()]), store=store, events=events)
    assert store.prunes == []
    _run(
        [_profile(2)],
        _Decider([_dislike()]),
        store=_PruningStore([]),
        auto_review_max_passes=0,
        auto_review_max_age_days=0,
    )  # 0 means unlimited, exactly like the config


def test_a_failed_archive_does_not_count_towards_the_sweep(monkeypatch):
    monkeypatch.setattr(worker_mod, "_AUTO_PRUNE_EVERY", 2)
    events: list = []
    store = _PruningStore(events, auto_pass=lambda **kw: False)
    _run(
        [_profile(i) for i in range(4)],
        _Decider([_dislike()]),
        store=store,
        events=events,
        auto_review_max_passes=5,
    )
    assert events.count("prune") == 1  # the run-start sweep only


def test_a_failing_prune_is_loud_and_never_stops_the_run(monkeypatch, capsys):
    monkeypatch.setattr(worker_mod, "_AUTO_PRUNE_EVERY", 2)

    def boom(**kw):
        raise RuntimeError("gcs unavailable")

    events: list = []
    store = _PruningStore(events, prune=boom)
    _, driver, store, events = _run(
        [_profile(i) for i in range(4)],
        _Decider([_dislike()]),
        store=store,
        events=events,
        auto_review_max_age_days=7,
    )

    assert (
        events.count("dislike") == 4 and len(store.auto_passes) == 4
    )  # run unaffected
    out = capsys.readouterr().out
    assert out.count("retention could not prune the archive") == 3  # start + 2 mid-run
    assert "RuntimeError: gcs unavailable" in out


def test_the_sweep_reports_what_it_removed(capsys):
    store = _PruningStore([], prune=lambda **kw: 12)
    _run([_profile(1)], _Decider([_dislike()]), store=store, auto_review_max_passes=400)
    assert "retention removed 12 archived pass(es)" in capsys.readouterr().out


def test_a_store_without_an_archive_is_not_an_error_for_retention():
    store = _StoreWithoutArchive([])
    _run(
        [_profile(1)],
        _Decider([_dislike()]),
        store=store,
        auto_review_max_passes=400,
        auto_review_max_age_days=30,
    )
    assert store.decisions == ["dislike"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"auto_review_max_passes": -1},
        {"auto_review_max_passes": True},
        {"auto_review_max_passes": 2.5},
        {"auto_review_max_passes": "400"},
        {"auto_review_max_age_days": -3},
        {"auto_review_max_age_days": True},
        {"auto_review_max_age_days": math.nan},
        {"auto_review_max_age_days": "30"},
    ],
)
def test_retention_bounds_are_validated_before_the_thread_touches_a_device(kwargs):
    with pytest.raises(ValueError):
        Worker(
            "bumble",
            _Driver([], []),
            _Decider([_dislike()]),
            None,
            _Store([]),
            "run-1",
            _Pacing(),
            threading.Event(),
            mode="auto",
            auto_review_random_share=0.5,
            **kwargs,
        )
