"""PreferenceModel tests — exercises the pure-Python LR fallback (no sklearn)."""
import math

import pytest

from operation_love.ranker.model import PreferenceModel

# Liveness bound, not a performance bound: it exists only so a genuine hang fails this test
# instead of hanging the whole suite forever. Widened 2026-08-22 when `python -m pytest` moved
# to one worker per core (pyproject.toml addopts `-n auto --dist loadgroup`), which measured a
# ~15x slowdown (0.33s idle vs 5.06s under load) on tests/test_concurrency.py's positive
# liveness waits of the same shape. Nothing about the property under test (did the reader/writer
# thread reach the expected state?) depends on the exact number, so widening it loses nothing.
_LIVENESS_TIMEOUT_S = 15.0


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
    m = PreferenceModel(min_labels=10, threshold=0.5)
    bad = [(True, [float("nan"), 2.0]) for _ in range(15)]
    bad += [(False, [-2.0, float("inf")]) for _ in range(15)]
    with pytest.raises(ValueError):
        m.train(bad)
    assert not m.ready


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"min_labels": True}, "min_labels"),
    ({"min_labels": 0}, "min_labels"),
    ({"min_labels": 2, "min_per_class": 0}, "min_per_class"),
    ({"min_labels": 2, "threshold": math.nan}, "threshold"),
    ({"min_labels": 2, "threshold": 10 ** 10_000}, "threshold"),
    ({"min_labels": 2, "threshold": True}, "threshold"),
])
def test_constructor_rejects_invalid_direct_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PreferenceModel(**kwargs)


@pytest.mark.parametrize("samples", [
    [(1, [1.0]), (False, [-1.0])],
    [(True, []), (False, [])],
    [(True, [1.0, 2.0]), (False, [-1.0])],
    [(True, [1.0, "2"]), (False, [-1.0, -2.0])],
    [(True, [1.0, math.nan]), (False, [-1.0, -2.0])],
    [(True, [1.0, 10 ** 10_000]), (False, [-1.0, -2.0])],
])
def test_training_rejects_malformed_labels_and_vectors_before_backend_selection(samples):
    model = PreferenceModel(min_labels=2, min_per_class=1)
    with pytest.raises(ValueError):
        model.train(samples)
    assert not model.ready


def test_pure_python_fallback_rejects_nonfinite_training_data(monkeypatch):
    import operation_love.ranker.model as model_mod

    monkeypatch.setattr(model_mod, "new_classifier",
                        lambda: (_ for _ in ()).throw(ImportError("no sklearn")))
    model = PreferenceModel(min_labels=2, min_per_class=1)
    with pytest.raises(ValueError, match="finite real"):
        model.train([(True, [math.nan]), (False, [-1.0])])
    assert not model.ready


@pytest.mark.parametrize("vector", [
    [1.0],
    [1.0, 2.0, 3.0],
    [1.0, math.inf],
    [1.0, True],
    [1.0, 10 ** 10_000],
])
def test_prediction_requires_the_trained_finite_feature_shape(vector):
    model = PreferenceModel(min_labels=2, min_per_class=1)
    assert model.train(_separable())
    with pytest.raises(ValueError):
        model.predict_proba(vector)


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


def test_failed_retrain_clears_a_previously_ready_classifier(monkeypatch):
    """A failed refresh must not leave stale preferences eligible for auto decisions."""
    import operation_love.ranker.model as model_mod

    class _BoomClf:
        def fit(self, X, y):
            raise ValueError("bad refreshed labels")

    model = PreferenceModel(min_labels=2)
    assert model.train(_separable()) is True
    assert model.ready

    monkeypatch.setattr(model_mod, "new_classifier", lambda: _BoomClf())
    import pytest
    with pytest.raises(ValueError, match="bad refreshed labels"):
        model.train(_separable())
    assert not model.ready
    with pytest.raises(RuntimeError, match="not ready"):
        model.predict_proba([2.0, 2.0])


def test_readers_wait_for_a_gated_retrain_then_see_the_published_model(monkeypatch):
    """A shared model must not expose the temporary cleared sentinel during fit()."""
    import threading

    import operation_love.ranker.model as model_mod

    fit_started = threading.Event()
    allow_fit_to_finish = threading.Event()

    class _ReadyClassifier:
        def fit(self, X, y):
            pass

        def predict_proba(self, rows):
            return [[0.1, 0.9] for _ in rows]

    class _GatedClassifier(_ReadyClassifier):
        def fit(self, X, y):
            fit_started.set()
            assert allow_fit_to_finish.wait(timeout=_LIVENESS_TIMEOUT_S)

    classifiers = iter((_ReadyClassifier(), _GatedClassifier()))
    monkeypatch.setattr(model_mod, "new_classifier", lambda: next(classifiers))

    model = PreferenceModel(min_labels=2)
    assert model.train(_separable()) is True

    retrain_result, retrain_errors = [], []

    def retrain():
        try:
            retrain_result.append(model.train(_separable()))
        except Exception as exc:  # pragma: no cover - assertion below reports it
            retrain_errors.append(exc)

    training = threading.Thread(target=retrain)
    training.start()
    assert fit_started.wait(timeout=_LIVENESS_TIMEOUT_S)

    ready_entered = threading.Event()
    predict_entered = threading.Event()
    ready_done = threading.Event()
    predict_done = threading.Event()
    observed = {}

    def read_ready():
        ready_entered.set()
        observed["ready"] = model.ready
        ready_done.set()

    def predict():
        predict_entered.set()
        observed["probability"] = model.predict_proba([2.0, 2.0])
        predict_done.set()

    ready_reader = threading.Thread(target=read_ready)
    prediction_reader = threading.Thread(target=predict)
    ready_reader.start()
    prediction_reader.start()
    assert (ready_entered.wait(timeout=_LIVENESS_TIMEOUT_S)
            and predict_entered.wait(timeout=_LIVENESS_TIMEOUT_S))

    try:
        assert not ready_done.wait(timeout=0.1)
        assert not predict_done.wait(timeout=0.1)
    finally:
        allow_fit_to_finish.set()
        training.join(timeout=_LIVENESS_TIMEOUT_S)
        ready_reader.join(timeout=_LIVENESS_TIMEOUT_S)
        prediction_reader.join(timeout=_LIVENESS_TIMEOUT_S)

    assert not training.is_alive()
    assert not ready_reader.is_alive()
    assert not prediction_reader.is_alive()
    assert retrain_errors == []
    assert retrain_result == [True]
    assert observed == {"ready": True, "probability": 0.9}
