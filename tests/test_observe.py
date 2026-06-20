"""Observe mode (shadow learning) — learn from manual swipes, retrain live.

Offline: a fake driver replays (profile, your_decision) pairs; a fake embedder
returns each profile's vector; the real PreferenceModel + RankerDecider + Worker
drive the learning loop.
"""
import threading

from operation_love.drivers.base import DatingAppDriver
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

    def open_session(self): self.opened = True
    def out_of_profiles(self): return self.i >= len(self.swipes)
    def current_profile(self): return self.swipes[self.i][0]
    def wait_for_decision(self, timeout=120.0):
        liked = self.swipes[self.i][1]; self.i += 1; return liked
    def next_profile(self): return None
    def like(self, opener=None): pass
    def dislike(self): pass
    def close(self): self.closed = True


class FakeStore:
    def __init__(self):
        self.labels, self.sources, self.decisions = [], [], []
    def add_label(self, run_id, app, liked, embedding, source="manual", **k):
        self.labels.append((liked, embedding)); self.sources.append(source)
    def load_labels(self): return list(self.labels)
    def record_decision(self, run_id, app, decision, score): self.decisions.append(decision)
    def label_count(self): return len(self.labels)
    def flush(self): pass
    def close(self): pass


class _Pacing:
    swipe_delay_s = 0.0


def _profile(vec):
    return Profile(photos=[b"x"], meta={"app": "bumble", "vec": vec})


def test_observe_learns_from_manual_swipes_and_becomes_ready():
    # You swipe: 3 likes on one cluster, 3 passes on another.
    swipes = []
    for _ in range(3):
        swipes.append((_profile([2.0, 2.0]), True))
        swipes.append((_profile([-2.0, -2.0]), False))

    model = PreferenceModel(min_labels=4, threshold=0.5)   # not ready at start
    assert not model.ready
    decider = RankerDecider(FakeQuality(), FakeEmbedder(), model)
    store = FakeStore()
    driver = FakeObservingDriver(swipes)

    w = Worker("bumble", driver, decider, opener_service=None, store=store, run_id="r",
               pacing=_Pacing(), stop_event=threading.Event(), mode="observe", retrain_every=2)
    w._observe_loop()

    assert len(store.labels) == 6                     # every manual swipe became a label
    assert all(s == "manual" for s in store.sources)  # tagged as manual
    assert len(store.decisions) == 6
    assert model.ready                                # retrained live -> now usable
    assert model.decide([2.0, 2.0])[0] == "like"      # learned your taste
    assert model.decide([-2.0, -2.0])[0] == "dislike"
    assert driver.opened and driver.closed


def test_observe_retrains_final_partial_batch():
    # If the session ends before retrain_every, the final labels should still
    # engage the model before shutdown instead of waiting for the next run.
    swipes = []
    for _ in range(3):
        swipes.append((_profile([2.0, 2.0]), True))
        swipes.append((_profile([-2.0, -2.0]), False))

    model = PreferenceModel(min_labels=4, threshold=0.5)
    decider = RankerDecider(FakeQuality(), FakeEmbedder(), model)
    store = FakeStore()
    w = Worker("bumble", FakeObservingDriver(swipes), decider, None, store, "r",
               _Pacing(), threading.Event(), mode="observe", retrain_every=10)
    w._observe_loop()

    assert len(store.labels) == 6
    assert model.ready
    assert model.decide([2.0, 2.0])[0] == "like"


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


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    sys.exit(1 if failed else 0)
