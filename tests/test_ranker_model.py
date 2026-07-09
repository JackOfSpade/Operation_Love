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


def test_min_per_class_defers_when_one_class_is_too_sparse():
    """Ranker must stay not-ready when one class has fewer than min_per_class examples,
    even if total label count exceeds min_labels and both classes are present."""
    m = PreferenceModel(min_labels=10, threshold=0.5, min_per_class=5)
    # 12 labels total, but only 2 likes — too few to train a reliable classifier.
    samples = [(True, [1.0, 0.0])] * 2 + [(False, [-1.0, 0.0])] * 10
    ready = m.train(samples)
    assert not ready
    assert not m.ready

    # With exactly 5 of each class (10 total, at the floor) it should now train.
    samples = [(True, [1.0, 0.0])] * 5 + [(False, [-1.0, 0.0])] * 5
    ready = m.train(samples)
    assert ready
    assert m.ready


def test_genuine_fit_failure_surfaces_and_stays_not_ready():
    # NaN in the feature vectors: sklearn's LogisticRegression.fit rejects this with a
    # ValueError. That genuine fit failure must surface loudly, NOT be swallowed into
    # the pure-Python fallback (which has no NaN guard). Needs sklearn installed —
    # without it the pure-Python path would happily fit NaN weights.
    import pytest
    pytest.importorskip("sklearn")
    m = PreferenceModel(min_labels=10, threshold=0.5)
    bad = [(True, [float("nan"), 2.0]) for _ in range(15)]
    bad += [(False, [-2.0, float("inf")]) for _ in range(15)]
    with pytest.raises(ValueError):
        m.train(bad)
    assert not m.ready


def test_train_falls_back_to_purepy_when_sklearn_unavailable(monkeypatch):
    # sklearn IS installed in CI/this sandbox, so despite this file's docstring,
    # the pure-Python fallback is otherwise never actually exercised here.
    import operation_love.ranker.model as model_mod
    monkeypatch.setattr(model_mod, "new_classifier",
                        lambda: (_ for _ in ()).throw(ImportError("no sklearn")))
    m = PreferenceModel(min_labels=2)
    assert m.train(_separable()) is True
    assert m._impl == "purepy"
    assert m.decide([2.0, 2.0])[0] == "like"


def test_train_propagates_a_real_fit_error_instead_of_silently_degrading(monkeypatch):
    # Bug fix: a genuine bug in sklearn's .fit() (bad shapes, NaN/inf, a version
    # incompatibility) must surface, not be silently swallowed as "no sklearn" and
    # papered over with the (worse, untuned) pure-Python fallback.
    import operation_love.ranker.model as model_mod

    class _BoomClf:
        def fit(self, X, y):
            raise ValueError("bad shapes")

    monkeypatch.setattr(model_mod, "new_classifier", lambda: _BoomClf())
    m = PreferenceModel(min_labels=2)
    try:
        m.train(_separable())
    except ValueError as e:
        assert "bad shapes" in str(e)
    else:
        raise AssertionError("expected the real .fit() error to propagate")


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
