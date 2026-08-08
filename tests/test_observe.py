"""Observe mode (shadow learning) — learn from manual swipes, retrain live.

Offline: a fake driver replays (profile, your_decision) pairs; a fake embedder
returns each profile's vector; the real PreferenceModel + RankerDecider + Worker
drive the learning loop.
"""
import threading

from operation_love.drivers.base import DatingAppDriver, DriverClosed
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import RankerDecider
from operation_love.ranker.model import PreferenceModel
from operation_love.worker import Worker


class FakeQuality:
    def filter(self, photos):
        return photos


class FakeEmbedder:
    def embed_profile(self, profile):
        return profile.meta.get("vec")


class FakeObservingDriver(DatingAppDriver):
    def __init__(self, swipes):           # swipes: list[(Profile, liked: bool)]
        self.swipes = list(swipes)
        self.i = 0
        self.opened = self.closed = False
        self.wait_timeouts = []

    def open_session(self): self.opened = True
    def out_of_profiles(self): return self.i >= len(self.swipes)
    def current_profile(self): return self.swipes[self.i][0]
    def wait_for_decision(self, timeout=120.0, should_stop=None):
        self.wait_timeouts.append(timeout)
        liked = self.swipes[self.i][1]; self.i += 1; return liked
    def render_busy(self, message=None): pass
    def next_profile(self): return None
    def like(self, opener=None): pass
    def dislike(self): pass
    def close(self): self.closed = True


class ClosingAfterSwipesDriver(FakeObservingDriver):
    def out_of_profiles(self):
        if self.i >= len(self.swipes):
            raise DriverClosed("browser closed")
        return False


class EmptyCaptureThenGoodDriver(FakeObservingDriver):
    def __init__(self, swipes):
        super().__init__(swipes)
        self.current_calls = 0
        self.busy_messages = []

    def current_profile(self):
        self.current_calls += 1
        if self.current_calls == 1:
            return Profile(photos=[], meta={"app": "bumble", "vec": None})
        return self.swipes[self.i][0]

    def render_busy(self, message=None):
        self.busy_messages.append(message)


class RecordingBusyDriver(FakeObservingDriver):
    def __init__(self, swipes):
        super().__init__(swipes)
        self.busy_messages = []

    def render_busy(self, message=None):
        self.busy_messages.append(message)


class FakeStore:
    def __init__(self):
        self.labels, self.profiles, self.sources, self.decisions = [], [], [], []
    def record_profile(self, run_id, app, profile_id, liked, source="manual", **k):
        self.profiles.append((profile_id, liked, k)); return True
    def add_label(self, run_id, app, liked, embedding, source="manual", profile_id="", **k):
        self.labels.append((liked, embedding, profile_id, k)); self.sources.append(source)
    def load_labels(self): return [(liked, embedding) for liked, embedding, _, _ in self.labels]
    def record_decision(self, run_id, app, decision, score, source="auto"):
        self.decisions.append((decision, source))
    def flush(self): pass
    def close(self): pass


class ArchiveFailingStore(FakeStore):
    """Image archiving fails for every profile (e.g. GCS unreachable)."""
    def record_profile(self, run_id, app, profile_id, liked, source="manual", **k):
        self.profiles.append((profile_id, liked, k)); return False


class _Pacing:
    swipe_delay_s = 0.0


def _profile(vec):
    return Profile(photos=[b"x"], meta={"app": "bumble", "vec": vec})


def _profile_with_metadata(vec):
    return Profile(
        photos=[b"one", b"two"],
        bio="bio text",
        prompts=[("Prompt", "Answer")],
        meta={"app": "bumble", "vec": vec},
    )


def test_observe_learns_from_manual_swipes_and_becomes_ready():
    # You swipe: 3 likes on one cluster, 3 passes on another.
    swipes = []
    for _ in range(3):
        swipes.append((_profile([2.0, 2.0]), True))
        swipes.append((_profile([-2.0, -2.0]), False))

    model = PreferenceModel(min_labels=4, threshold=0.5, min_per_class=1)   # not ready at start
    assert not model.ready
    decider = RankerDecider(FakeQuality(), FakeEmbedder(), model)
    store = FakeStore()
    driver = FakeObservingDriver(swipes)

    w = Worker("bumble", driver, decider, opener_service=None, store=store, run_id="r",
               pacing=_Pacing(), stop_event=threading.Event(), mode="observe", retrain_every=2)
    w._observe_loop()

    assert len(store.labels) == 6                     # every manual swipe became a label
    assert len(store.profiles) == 6
    assert {label[2] for label in store.labels} == {profile[0] for profile in store.profiles}
    assert all(s == "manual" for s in store.sources)  # tagged as manual
    assert len(store.decisions) == 6
    assert all(source == "manual" for _, source in store.decisions)
    assert model.ready                                # retrained live -> now usable
    assert model.decide([2.0, 2.0])[0] == "like"      # learned your taste
    assert model.decide([-2.0, -2.0])[0] == "dislike"
    assert driver.opened and driver.closed
    assert driver.wait_timeouts == [None] * 6


def test_observe_logs_profile_text_separators(capsys):
    model = PreferenceModel(min_labels=10)
    decider = RankerDecider(FakeQuality(), FakeEmbedder(), model)
    store = FakeStore()
    driver = FakeObservingDriver([(_profile([1.0, 1.0]), True)])
    w = Worker("bumble", driver, decider, None, store, "r",
               _Pacing(), threading.Event(), mode="observe", retrain_every=5)

    w._observe_loop()

    out = capsys.readouterr().out
    assert f"\n{'-' * 72}\n" in out
    assert "✅ READY — swipe this profile" in out
    assert "Got LIKE — processing" in out
    assert "[worker-bumble]" not in out
    assert "profile #1" not in out


def test_observe_retrains_final_partial_batch():
    # If the session ends before retrain_every, the final labels should still
    # engage the model before shutdown instead of waiting for the next run.
    swipes = []
    for _ in range(3):
        swipes.append((_profile([2.0, 2.0]), True))
        swipes.append((_profile([-2.0, -2.0]), False))

    model = PreferenceModel(min_labels=4, threshold=0.5, min_per_class=1)
    decider = RankerDecider(FakeQuality(), FakeEmbedder(), model)
    store = FakeStore()
    w = Worker("bumble", FakeObservingDriver(swipes), decider, None, store, "r",
               _Pacing(), threading.Event(), mode="observe", retrain_every=10)
    w._observe_loop()

    assert len(store.labels) == 6
    assert model.ready
    assert model.decide([2.0, 2.0])[0] == "like"


def test_observe_retrains_final_partial_batch_on_browser_close():
    swipes = []
    for _ in range(3):
        swipes.append((_profile([2.0, 2.0]), True))
        swipes.append((_profile([-2.0, -2.0]), False))

    model = PreferenceModel(min_labels=4, threshold=0.5, min_per_class=1)
    decider = RankerDecider(FakeQuality(), FakeEmbedder(), model)
    store = FakeStore()
    driver = ClosingAfterSwipesDriver(swipes)
    w = Worker("bumble", driver, decider, None, store, "r",
               _Pacing(), threading.Event(), mode="observe", retrain_every=10)

    try:
        w._observe_loop()
    except DriverClosed:
        pass
    else:
        raise AssertionError("expected DriverClosed")

    assert len(store.labels) == 6
    assert model.ready
    assert driver.closed


def test_observe_skips_no_face():
    # A card the embedder can't embed (no face) must not become a label.
    swipes = [(_profile(None), True), (_profile([1.0, 1.0]), True)]
    model = PreferenceModel(min_labels=10)
    decider = RankerDecider(FakeQuality(), FakeEmbedder(), model)
    store = FakeStore()
    w = Worker("bumble", FakeObservingDriver(swipes), decider, None, store, "r",
               _Pacing(), threading.Event(), mode="observe", retrain_every=5)
    w._observe_loop()
    assert len(store.labels) == 1                     # the no-face card was skipped
    assert len(store.profiles) == 2                   # raw profile decisions remain replayable


def test_observe_recaptures_before_decision_when_profile_has_no_photos():
    swipes = [(_profile([1.0, 1.0]), True)]
    model = PreferenceModel(min_labels=10)
    decider = RankerDecider(FakeQuality(), FakeEmbedder(), model)
    store = FakeStore()
    driver = EmptyCaptureThenGoodDriver(swipes)
    w = Worker("bumble", driver, decider, None, store, "r",
               _Pacing(), threading.Event(), mode="observe", retrain_every=5)
    w._observe_loop()

    assert driver.current_calls == 2
    assert driver.busy_messages[0].startswith("Capturing profile")
    assert sum(1 for msg in driver.busy_messages if msg and msg.startswith("Capturing profile")) == 2
    assert None in driver.busy_messages                    # only unblocks after the good capture
    assert len(store.profiles) == 1
    assert len(store.labels) == 1
    assert len(store.decisions) == 1


def test_observe_reblocks_capture_after_decision_wait_returns_none():
    swipes = [(_profile([1.0, 1.0]), None), (_profile([2.0, 2.0]), True)]
    model = PreferenceModel(min_labels=10)
    decider = RankerDecider(FakeQuality(), FakeEmbedder(), model)
    store = FakeStore()
    driver = RecordingBusyDriver(swipes)
    w = Worker("bumble", driver, decider, None, store, "r",
               _Pacing(), threading.Event(), mode="observe", retrain_every=5)
    w._observe_loop()

    assert sum(1 for msg in driver.busy_messages if msg and msg.startswith("Capturing profile")) >= 2
    assert len(store.profiles) == 1
    assert len(store.labels) == 1


def test_observe_skips_label_when_image_archive_fails():
    # If a profile's images can't be archived, we keep no label without them — but
    # the observe session must NOT crash/restart over it; it just moves on.
    swipes = [(_profile([1.0, 1.0]), True), (_profile([2.0, 2.0]), False)]
    model = PreferenceModel(min_labels=10)
    decider = RankerDecider(FakeQuality(), FakeEmbedder(), model)
    store = ArchiveFailingStore()
    driver = FakeObservingDriver(swipes)
    w = Worker("bumble", driver, decider, None, store, "r",
               _Pacing(), threading.Event(), mode="observe", retrain_every=5)
    w._observe_loop()

    assert len(store.profiles) == 2          # archiving was attempted for each swipe
    assert len(store.labels) == 0            # but no label kept without its images
    assert len(store.decisions) == 0         # and the skipped swipe records no decision
    assert driver.closed                     # clean shutdown, not a crash/restart


def test_observe_label_persists_profile_metadata():
    model = PreferenceModel(min_labels=10)
    decider = RankerDecider(FakeQuality(), FakeEmbedder(), model)
    store = FakeStore()
    w = Worker("bumble", FakeObservingDriver([(_profile_with_metadata([1.0]), True)]),
               decider, None, store, "r", _Pacing(), threading.Event(),
               mode="observe", retrain_every=5)
    w._observe_loop()

    metadata = store.labels[0][3]
    assert metadata == {"photo_count": 2}
    assert store.profiles[0][2]["photo_count"] == 2
    assert "bio" not in store.profiles[0][2]
    assert "prompts" not in store.profiles[0][2]
