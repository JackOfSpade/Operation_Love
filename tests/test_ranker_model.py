"""PreferenceModel tests — exercises the pure-Python LR fallback (no sklearn)."""
from operation_love.ranker.model import PreferenceModel


def _separable(n=30):
    liked = [(True, [2.0, 2.0]) for _ in range(n)]
    disliked = [(False, [-2.0, -2.0]) for _ in range(n)]
    return liked + disliked


def test_learns_separable_preference():
    m = PreferenceModel(min_labels=10, threshold=0.5)
    assert m.train(_separable()) is True and m.ready
    assert m.decide([2.0, 2.0])[0] == "like"
    assert m.decide([-2.0, -2.0])[0] == "dislike"
    assert 0.0 <= m.predict_proba([2.0, 2.0]) <= 1.0


def test_cold_start_not_ready_below_min_labels():
    m = PreferenceModel(min_labels=100, threshold=0.5)
    assert m.train(_separable(5)) is False
    assert not m.ready


def test_needs_both_classes():
    m = PreferenceModel(min_labels=2, threshold=0.5)
    assert m.train([(True, [1.0, 1.0]), (True, [2.0, 2.0])]) is False
    assert not m.ready


def test_threshold_respected():
    m = PreferenceModel(min_labels=2, threshold=0.99)  # very strict
    m.train(_separable())
    # a borderline-positive point should fail a 0.99 bar -> dislike
    assert m.decide([0.2, 0.2])[0] == "dislike"


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
