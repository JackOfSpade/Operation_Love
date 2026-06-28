"""RankerDecider composition tests — no_face / cold-start defer / like-dislike."""
from operation_love.perception.capture import Profile
from operation_love.ranker.decider import RankerDecider


class FakeQuality:
    def filter(self, photos):
        return photos


class FakeEmbedder:
    def __init__(self, vec):
        self.vec = vec
    def embed_profile(self, profile):
        return self.vec


class FakeModel:
    def __init__(self, ready, decision=("like", 0.8)):
        self._ready = ready
        self._decision = decision
    @property
    def ready(self):
        return self._ready
    def decide(self, vec):
        return self._decision


def _profile():
    return Profile(photos=[b"x"], bio="hi")


def test_no_face_returns_no_face():
    d = RankerDecider(FakeQuality(), FakeEmbedder(None), FakeModel(True)).decide(_profile())
    assert d.decision == "no_face" and d.embedding == []


def test_cold_start_defers():
    d = RankerDecider(FakeQuality(), FakeEmbedder([1.0, 2.0]), FakeModel(False)).decide(_profile())
    assert d.decision == "defer" and d.source == "cold_start" and d.embedding == [1.0, 2.0]


def test_ranker_likes():
    model = FakeModel(True, ("like", 0.91))
    d = RankerDecider(FakeQuality(), FakeEmbedder([1.0, 2.0, 3.0]), model).decide(_profile())
    assert d.decision == "like" and d.score == 0.91 and d.embedding == [1.0, 2.0, 3.0]


def test_ranker_dislikes():
    model = FakeModel(True, ("dislike", 0.12))
    d = RankerDecider(FakeQuality(), FakeEmbedder([0.0]), model).decide(_profile())
    assert d.decision == "dislike" and d.score == 0.12


class EmptyQuality:
    def filter(self, photos):
        return []


class SpyEmbedder:
    def __init__(self, vec):
        self.vec = vec
        self.seen = None
    def embed_profile(self, profile):
        self.seen = profile.photos
        return self.vec


def test_empty_filter_falls_back_to_original_photos():
    embedder = SpyEmbedder([7.0, 8.0])
    profile = _profile()
    vec = RankerDecider(EmptyQuality(), embedder, FakeModel(True)).embed(profile)
    assert embedder.seen == profile.photos and vec == [7.0, 8.0]


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
