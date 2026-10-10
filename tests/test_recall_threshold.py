"""Recall-target threshold: ranker/threshold.py, the recall-target PreferenceModel, the
decider's retrain wiring and the supervisor's startup wiring.

Every guard here has a fixture that REACHES the branch it is named for (see the repo's
"fixtures can miss the branch they name" lesson): where a test could pass vacuously the test
first asserts the precondition that makes the guard load-bearing (for example that including
the review labels WOULD have changed the threshold, or that the naive float arithmetic WOULD
have rounded the needed count up).
"""

import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import operation_love.ranker.model as model_mod
import operation_love.supervisor as sup
from operation_love import interaction
from operation_love.drivers.base import DatingAppDriver
from operation_love.ranker import threshold as thr
from operation_love.ranker.decider import RankerDecider
from operation_love.ranker.evaluate import identity_groups
from operation_love.ranker.model import PreferenceModel

_DIM = 1280  # production width: 512 ArcFace dims + 768 CLIP dims
_FACE = 512


def _dataset(n, seed=1, signal=0.05, like_rate=0.4):
    """Production-shaped rows whose out-of-fold scores are COMPRESSED around 0.5.

    The first 512 dims are a random unit "face" per row (so every row is its own identity under
    the evaluator's cosine DBSCAN); the label signal is one tail dimension.  At signal=0.05 the
    out-of-fold scores span ~0.47-0.51, the shape of the real ranker, so a fixed 0.5 cutoff
    passes most of the people the owner would like.
    """
    rng = np.random.default_rng(seed)
    face = rng.normal(size=(n, _FACE))
    face /= np.linalg.norm(face, axis=1, keepdims=True)
    tail = rng.normal(size=(n, _DIM - _FACE)) / math.sqrt(_DIM - _FACE)
    y = rng.random(n) < like_rate
    tail[:, 0] += signal * (2 * y - 1)
    return [bool(v) for v in y], np.concatenate([face, tail], axis=1)


def _pairs(y, X):
    return [(liked, x.tolist()) for liked, x in zip(y, X, strict=True)]


def _oof(y, X):
    groups = identity_groups(X[:, :_FACE].tolist())
    return thr.oof_scores(X, [int(v) for v in y], groups)


# --------------------------------------------------------------------------------------------
# recall_threshold
# --------------------------------------------------------------------------------------------


def test_recall_threshold_takes_the_finite_sample_index_of_the_descending_scores():
    scores = [i / 100 for i in range(1, 21)]  # 0.01 .. 0.20, n=20
    # ceil(0.5 * 21) = 11 positives must be kept: the 11th highest is 0.10.  The in-sample
    # index ceil(0.5 * 20) = 10 (0.11) delivers only 10/21 = 0.476 of a FRESH positive stream.
    assert thr.recall_threshold(scores, 0.5) == pytest.approx(0.10)
    # ceil(0.9 * 21) = 19 -> the 19th highest.
    assert thr.recall_threshold(scores, 0.9) == pytest.approx(0.02)
    # ceil(0.95 * 21) = 20 -> every positive: the target needs a 21st score nobody has, so
    # the cutoff is the minimum (the old in-sample index answered 0.02 here).
    assert thr.recall_threshold(scores, 0.95) == pytest.approx(0.01)
    # Order of the input is irrelevant.
    assert thr.recall_threshold(scores[::-1], 0.5) == pytest.approx(0.10)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("target", [0.5, 0.8, 0.9, 0.95, 0.99])
def test_recall_threshold_is_the_highest_cutoff_that_keeps_the_needed_count(
    seed, target
):
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.2, 0.8, size=int(rng.integers(5, 80))).tolist()
    t = thr.recall_threshold(scores, target)
    n = len(scores)
    needed = min(n, math.ceil(target * (n + 1) - 1e-9))
    assert sum(s >= t for s in scores) >= needed
    higher = [s for s in scores if s > t]
    if higher:
        # "highest": the next cutoff up already keeps too few.
        assert sum(s >= min(higher) for s in scores) < needed


def test_recall_threshold_float_noise_does_not_buy_unrequested_recall():
    # 0.07 * 100 is 7.000000000000001 in binary floating point, so a bare ceil() asks for 8
    # positives.  Pin the precondition so the test cannot pass vacuously on a platform where
    # the product happens to be exact.  n=99 makes the needed count t * (n + 1) exactly 7.
    assert math.ceil(0.07 * 100) == 8
    scores = [i / 1000 for i in range(1, 100)]  # n=99, desc: 0.099, 0.098, ...
    assert thr.recall_threshold(scores, 0.07) == pytest.approx(
        0.093
    )  # the 7th, not 8th


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@pytest.mark.parametrize("n", [19, 20, 21, 40, 100])
def test_recall_threshold_delivers_the_target_on_fresh_positives(n):
    """The cutoff is read from n sample scores but judged on the people who come NEXT.

    i.i.d. N(0,1) positive scores, so the true recall of a cutoff c is exactly 1 - Phi(c) and
    can be averaged over many samples without simulating the fresh stream.  The in-sample
    quantile index ceil(0.95 * n) delivered 0.906 at n=20, 0.927 at n=40 and 0.941 at n=100
    (expected recall k/(n+1)); the finite-sample index holds 0.95.  Mutating the (n + 1) back to
    n fails n=20 and n=40 by a wide margin.
    """
    rng = np.random.default_rng(1000 + n)
    trials = 1500
    total = 0.0
    for _ in range(trials):
        sample = rng.normal(size=n).tolist()
        # scores are standard-normal here, so bypass the probability clamp with a shift that
        # keeps every draw inside (0, 1): the transform is monotone, recall is unchanged.
        cutoff = thr.recall_threshold([0.5 + s / 10.0 for s in sample], 0.95)
        total += 1.0 - _normal_cdf((cutoff - 0.5) * 10.0)
    mean_recall = total / trials
    assert mean_recall >= 0.945, mean_recall
    assert mean_recall <= 0.99  # and it is not simply "keep everyone" either


@pytest.mark.parametrize(
    "target,expected",
    [(0.5, 1), (0.8, 4), (0.9, 9), (0.95, 19), (0.99, 99), (0.999, 999)],
)
def test_min_positives_for_target_is_where_the_target_becomes_measurable(
    target, expected
):
    assert thr.min_positives_for_target(target) == expected
    # Below it the needed count would exceed n (clamped to the minimum score); at it, it does not.
    below = [i / 10_000 + 0.1 for i in range(expected - 1)]
    at = [i / 10_000 + 0.1 for i in range(expected)]
    if below:
        assert math.ceil(target * (len(below) + 1) - 1e-9) > len(below)
    assert math.ceil(target * (len(at) + 1) - 1e-9) <= len(at)
    for bad in (0.0, 1.0, math.nan, True, "0.9"):
        with pytest.raises(ValueError):
            thr.min_positives_for_target(bad)


def test_recall_threshold_clamps_into_a_usable_probability():
    assert thr.recall_threshold([0.9999, 0.99995], 0.5) == thr.MAX_THRESHOLD
    assert thr.recall_threshold([0.0, 0.0], 0.5) == thr.MIN_THRESHOLD
    assert 0.0 < thr.MIN_THRESHOLD < thr.MAX_THRESHOLD < 1.0


@pytest.mark.parametrize(
    "scores,target",
    [
        ([], 0.9),
        ([0.5], 0.0),
        ([0.5], 1.0),
        ([0.5], -0.1),
        ([0.5], True),
        ([0.5], math.nan),
        ([0.5], math.inf),
        ([0.5], "0.9"),
        ([math.nan], 0.9),
        ([0.5, math.inf], 0.9),
        ([0.5, True], 0.9),
    ],
)
def test_recall_threshold_rejects_degenerate_input_loudly(scores, target):
    with pytest.raises(ValueError):
        thr.recall_threshold(scores, target)


# --------------------------------------------------------------------------------------------
# wilson_lower / live adjustment
# --------------------------------------------------------------------------------------------


def test_wilson_lower_matches_the_closed_form_and_is_conservative():
    assert thr.wilson_lower(0, 0) == 0.0
    # Hand-computed: p=.5, n=10, z=1.645 -> (0.6353 - 0.2932) / 1.2706 = 0.2692.
    assert thr.wilson_lower(5, 10) == pytest.approx(0.2692, abs=1e-3)
    assert thr.wilson_lower(0, 10) == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < thr.wilson_lower(10, 10) < 1.0
    # Lower than the point estimate, tighter with more evidence at the same rate, monotone in k.
    assert thr.wilson_lower(5, 10) < thr.wilson_lower(50, 100) < 0.5
    assert thr.wilson_lower(4, 10) < thr.wilson_lower(5, 10)


@pytest.mark.parametrize(
    "k,n,z",
    [(-1, 5, 1.645), (6, 5, 1.645), (True, 5, 1.645), (1, 5.0, 1.645), (1, 5, -1.0)],
)
def test_wilson_lower_rejects_impossible_counts(k, n, z):
    with pytest.raises(ValueError):
        thr.wilson_lower(k, n, z)


def _live(random_reviewed=40, random_would_like=30, auto_likes=40, auto_passes=60):
    return {
        "random_reviewed": random_reviewed,
        "random_would_like": random_would_like,
        "auto_likes": auto_likes,
        "auto_passes": auto_passes,
    }


def test_live_adjustment_is_the_documented_formula():
    out = thr.live_adjustment(_live(), expected_false_pass_rate=0.10, base_rate=0.4)
    # Significant (the Wilson lower bound is above f_exp), so the target moves by the POINT
    # estimate of the gap: (k/n - f_exp) * pass_share / base_rate.
    assert thr.wilson_lower(30, 40) > 0.10
    want = (30 / 40 - 0.10) * 0.6 / 0.4
    assert out["extra"] == pytest.approx(want) and want > 0
    assert out["pass_share"] == pytest.approx(0.6)
    assert out["observed_false_pass_lower"] == pytest.approx(thr.wilson_lower(30, 40))
    assert (out["random_reviewed"], out["random_would_like"]) == (40, 30)


def test_live_adjustment_wilson_bound_is_a_gate_not_a_discount():
    """An observed rate above f_exp that is NOT significant moves nothing; a significant one
    moves the target by the whole measured gap, not the gap minus the confidence margin.

    n=200, f_exp=0.20, pass_share/base_rate = 0.5 (0.15 / 0.30): a real recall shortfall of 0.03
    is a false-pass gap of 0.06, i.e. k=52 of 200 (26%).  The first version subtracted the
    Wilson margin before scaling, so it asked for only (0.209 - 0.20) * 0.5 = 0.0045 of the 0.03
    that was missing -- a dead zone of ~0.02 recall.  Mutating the gate back into a Wilson
    subtraction fails the second assertion; dropping the gate fails the first.
    """
    live = dict(random_reviewed=200, random_would_like=0, auto_likes=85, auto_passes=15)
    insignificant = thr.live_adjustment(
        {**live, "random_would_like": 46},  # 23% > 20%, but the lower bound is not
        expected_false_pass_rate=0.20,
        base_rate=0.30,
    )
    assert (
        46 / 200 > 0.20 > thr.wilson_lower(46, 200)
    )  # precondition: noise, not evidence
    assert insignificant["extra"] == 0.0

    significant = thr.live_adjustment(
        {**live, "random_would_like": 52},
        expected_false_pass_rate=0.20,
        base_rate=0.30,
    )
    assert thr.wilson_lower(52, 200) > 0.20  # precondition: the gate opens
    shortfall = (52 / 200 - 0.20) * 0.15 / 0.30
    assert significant["extra"] == pytest.approx(shortfall)
    assert (
        significant["extra"] > 0.9 * 0.03
    )  # recovers the real 0.03 shortfall in one step
    assert significant["extra"] > 4 * (thr.wilson_lower(52, 200) - 0.20) * 0.5


def test_live_adjustment_counts_only_model_passes_in_the_pass_share():
    """Likes the session policy or the like-ratio ceiling demoted to a pass are 'dislike'
    decisions the random pool never samples; counting them inflated the share of passes the
    measured false-pass rate applies to."""
    base = _live(auto_likes=40, auto_passes=60)
    inflated = thr.live_adjustment(base, expected_false_pass_rate=0.10, base_rate=0.4)
    model_only = thr.live_adjustment(
        {**base, "auto_model_passes": 30},  # 30 genuine passes + 30 demoted likes
        expected_false_pass_rate=0.10,
        base_rate=0.4,
    )
    assert inflated["pass_share"] == pytest.approx(0.6)  # a store without the field
    assert model_only["pass_share"] == pytest.approx(0.3)  # 30 of 100 decisions
    assert model_only["extra"] == pytest.approx(inflated["extra"] / 2)
    for bad in (-1, True, 1.5, 61):  # negative, bool, float, more than auto_passes
        with pytest.raises(ValueError):
            thr.live_adjustment(
                {**base, "auto_model_passes": bad},
                expected_false_pass_rate=0.1,
                base_rate=0.4,
            )
    # An explicit None is the same as absent (a store that reports the key without a value).
    same = thr.live_adjustment(
        {**base, "auto_model_passes": None},
        expected_false_pass_rate=0.10,
        base_rate=0.4,
    )
    assert same["pass_share"] == pytest.approx(0.6)


def test_live_adjustment_needs_ten_random_answers():
    # Same enormous gap; only the sample size differs.  Mutating the n >= 10 guard out makes the
    # 9-answer case non-zero.
    nine = thr.live_adjustment(
        _live(random_reviewed=9, random_would_like=9),
        expected_false_pass_rate=0.0,
        base_rate=0.4,
    )
    ten = thr.live_adjustment(
        _live(random_reviewed=10, random_would_like=10),
        expected_false_pass_rate=0.0,
        base_rate=0.4,
    )
    assert nine["extra"] == 0.0 and nine["observed_false_pass_lower"] > 0
    assert ten["extra"] > 0


def test_live_adjustment_never_lowers_the_target():
    below = thr.live_adjustment(
        _live(random_reviewed=40, random_would_like=2),
        expected_false_pass_rate=0.3,
        base_rate=0.4,
    )
    assert below["extra"] == 0.0
    no_decisions = thr.live_adjustment(
        _live(auto_likes=0, auto_passes=0),
        expected_false_pass_rate=0.0,
        base_rate=0.4,
    )
    assert no_decisions["extra"] == 0.0 and no_decisions["pass_share"] == 0.0


def test_live_adjustment_none_and_malformed():
    assert (
        thr.live_adjustment(None, expected_false_pass_rate=0.1, base_rate=0.4) is None
    )
    for bad in (
        "nope",
        {"random_reviewed": 5},
        {**_live(), "random_would_like": 99},
        {**_live(), "auto_passes": -1},
        {**_live(), "auto_likes": True},
        {**_live(), "random_reviewed": 4.0},
    ):
        with pytest.raises(ValueError):
            thr.live_adjustment(bad, expected_false_pass_rate=0.1, base_rate=0.4)


def test_effective_target_is_capped_below_like_everyone():
    assert thr.effective_target(0.95, 0.0) == 0.95
    assert thr.effective_target(0.95, 0.02) == pytest.approx(0.97)
    assert thr.effective_target(0.95, 5.0) == thr.MAX_EFFECTIVE_TARGET == 0.99
    assert thr.effective_target(0.95, -1.0) == 0.95  # a negative correction is ignored


def test_effective_target_never_lowers_a_target_the_owner_configured_above_the_cap():
    # The 0.99 cap bounds the live INCREMENT.  The first version applied it to the configured
    # target as well, so `like_recall_target: 0.995` silently became 0.99.
    assert thr.effective_target(0.995, 0.0) == 0.995
    assert (
        thr.effective_target(0.995, 0.01) == 0.995
    )  # no live raise past the owner's own
    assert thr.effective_target(0.999, 5.0) == 0.999
    assert thr.effective_target(0.98, 5.0) == thr.MAX_EFFECTIVE_TARGET


def test_calibration_label_sources():
    assert thr.is_calibration_source(None)
    assert thr.is_calibration_source("")
    assert thr.is_calibration_source("hub_training")
    assert not thr.is_calibration_source("auto_review_random")
    assert not thr.is_calibration_source("auto_review_near_cut")
    assert not thr.is_calibration_source("auto_review")
    with pytest.raises(ValueError):
        thr.is_calibration_source(7)


# --------------------------------------------------------------------------------------------
# oof_scores
# --------------------------------------------------------------------------------------------


def _auc(y, scores):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, scores)


def test_oof_scores_honours_identity_groups():
    """Pure-noise features, but every person appears three times with the SAME label and the
    SAME vector.  Honouring the groups keeps a person out of their own training fold, so the
    out-of-fold AUC is chance; ignoring them (every row its own group) lets the classifier
    memorise the duplicate and scores near-perfectly.  The second assertion is the control that
    proves the first one can fail."""
    rng = np.random.default_rng(3)
    people = 40
    face = rng.normal(size=(people, _FACE))
    face /= np.linalg.norm(face, axis=1, keepdims=True)
    tail = rng.normal(size=(people, _DIM - _FACE)) / math.sqrt(_DIM - _FACE)
    labels = np.array([i % 2 for i in range(people)])
    X = np.repeat(np.concatenate([face, tail], axis=1), 3, axis=0)
    y = np.repeat(labels, 3)
    person = np.repeat(np.arange(people), 3).tolist()

    grouped = thr.oof_scores(X, y, person)
    assert grouped is not None and len(grouped) == len(y)
    assert ((grouped >= 0) & (grouped <= 1)).all()
    assert _auc(y, grouped) < 0.7

    leaked = thr.oof_scores(X, y, list(range(len(y))))
    assert _auc(y, leaked) > 0.9

    # The default grouping (cosine DBSCAN on the first 512 dims) must find the same people.
    assert _auc(y, thr.oof_scores(X, y)) < 0.7


def test_oof_scores_none_when_fewer_than_two_folds_are_possible():
    y_all, X = _dataset(30, seed=4)
    y = [int(v) for v in y_all]
    assert thr.oof_scores(X, y, [0] * 30) is None  # one identity
    one_like = [1] + [0] * 29
    assert (
        thr.oof_scores(X, one_like, list(range(30))) is None
    )  # one positive, one fold


def test_oof_scores_none_rather_than_partial_when_a_training_side_lacks_a_class():
    # Every positive belongs to ONE identity, so the fold that holds it out trains on passes
    # only.  Its rows cannot be scored out-of-fold; a vector with holes (or a threshold read off
    # the surviving subset) would silently calibrate on a biased sample, so the answer is None.
    _, X = _dataset(30, seed=5)
    y = [1] * 6 + [0] * 24
    groups = [0] * 6 + list(range(1, 25))
    assert thr.oof_scores(X, y, groups) is None


def test_oof_scores_rejects_malformed_input():
    _, X = _dataset(20, seed=6)
    y = [i % 2 for i in range(20)]
    with pytest.raises(ValueError):
        thr.oof_scores(X, y, list(range(19)))  # group count mismatch
    with pytest.raises(ValueError):
        thr.oof_scores(X, y[:-1], list(range(20)))  # label count mismatch
    with pytest.raises(ValueError):
        thr.oof_scores(X, [2] + y[1:], list(range(20)))  # not 0/1
    with pytest.raises(ValueError):
        thr.oof_scores(X, y, list(range(20)), n_splits=1)
    with pytest.raises(ValueError):
        thr.oof_scores(X, y, list(range(20)), n_splits=True)


# --------------------------------------------------------------------------------------------
# PreferenceModel -- construction
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        0,
        0.0,
        1,
        1.0,
        -0.1,
        1.5,
        math.nan,
        math.inf,
        True,
        "0.9",
        pytest.param(10**10_000, id="huge-int"),
    ],
)
def test_recall_target_must_be_none_or_strictly_inside_unit_interval(bad):
    with pytest.raises(ValueError, match="recall_target"):
        PreferenceModel(10, recall_target=bad)


@pytest.mark.parametrize("bad", [-0.01, 0.21, math.nan, math.inf, True, "0.02", None])
def test_contextual_recall_budget_bounds(bad):
    with pytest.raises(ValueError, match="contextual_recall_budget"):
        PreferenceModel(10, recall_target=0.9, contextual_recall_budget=bad)


def test_boundary_values_construct():
    PreferenceModel(10, recall_target=0.001, contextual_recall_budget=0.0)
    PreferenceModel(10, recall_target=0.999, contextual_recall_budget=0.2)
    PreferenceModel(10, recall_target=None)


def test_default_max_lift_mirrors_the_interaction_policy_ceiling():
    # model.py cannot import interaction (import cycle), so the constant is mirrored.
    assert model_mod.DEFAULT_MAX_THRESHOLD_LIFT == interaction._THRESHOLD_MAX_LIFT


# --------------------------------------------------------------------------------------------
# PreferenceModel -- legacy behaviour is untouched
# --------------------------------------------------------------------------------------------


def test_legacy_model_keeps_its_fixed_threshold_and_ignores_sources():
    y, X = _dataset(60, seed=7, signal=0.4)
    pairs = _pairs(y, X)
    # Sources that WOULD change calibration under a recall target; a legacy model must not care.
    triples = [(liked, vec, "auto_review_random") for liked, vec in pairs]
    a = PreferenceModel(40, threshold=0.37)
    b = PreferenceModel(40, threshold=0.37)
    assert a.train(pairs) is True and b.train(triples) is True
    assert a.threshold == b.threshold == 0.37
    assert a.calibration is None and b.calibration is None
    assert a.max_threshold_lift == 0.045
    probe = pairs[0][1]
    assert a.predict_proba(probe) == pytest.approx(b.predict_proba(probe), abs=1e-12)
    assert a.not_ready_reason is None


def test_legacy_live_review_is_ignored_even_when_malformed():
    y, X = _dataset(60, seed=7, signal=0.4)
    m = PreferenceModel(40)
    assert m.train(_pairs(y, X), live_review={"garbage": 1}) is True
    assert m.threshold == 0.5


def test_legacy_not_ready_reasons_are_reported():
    m = PreferenceModel(100)
    assert m.not_ready_reason == "not trained yet"
    assert m.train([(True, [1.0, 1.0]), (False, [-1.0, -1.0])]) is False
    assert "min_labels" in m.not_ready_reason
    m = PreferenceModel(2)
    assert m.train([(True, [1.0]), (True, [2.0])]) is False
    assert "one class" in m.not_ready_reason
    m = PreferenceModel(2, min_per_class=5)
    assert m.train([(True, [1.0])] * 2 + [(False, [-1.0])] * 10) is False
    assert "min_per_class" in m.not_ready_reason


def test_sample_shape_validation_accepts_triples_and_rejects_the_rest():
    m = PreferenceModel(2, min_per_class=1)
    good = [(True, [1.0, 1.0], None), (False, [-1.0, -1.0], "")]
    assert m.train(good) is True
    for bad in (
        [(True, [1.0], "x", "extra"), (False, [-1.0], "x", "extra")],
        [(True, [1.0], 7), (False, [-1.0], 7)],
        [(True,), (False,)],
    ):
        with pytest.raises(ValueError):
            m.train(bad)
        assert not m.ready


# --------------------------------------------------------------------------------------------
# PreferenceModel -- recall-target calibration
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compressed():
    y, X = _dataset(120, seed=1, signal=0.05)
    return y, X, _pairs(y, X)


def test_recall_target_sets_the_threshold_that_holds_recall_on_oof_scores(compressed):
    y, X, rows = compressed
    oof = _oof(y, X)
    pos = [s for s, liked in zip(oof, y, strict=True) if liked]
    # Precondition: the shipped fixed cutoff really does lose most of the likes here.  Without
    # this the test could pass on data where 0.5 already satisfied the recall target.
    assert np.mean(np.array(pos) >= 0.5) < 0.5

    m = PreferenceModel(40, threshold=0.5, recall_target=0.9)
    assert m.train(rows) is True and m.ready and m.not_ready_reason is None
    assert m.threshold == pytest.approx(thr.recall_threshold(pos, 0.9), rel=1e-9)
    assert m.threshold < 0.5
    assert np.mean(np.array(pos) >= m.threshold) >= 0.9
    cal = m.calibration
    assert cal["threshold"] == m.threshold
    assert cal["recall_target"] == cal["effective_target"] == 0.9
    assert cal["calibration_labels"] == 120 and cal["review_labels"] == 0
    assert cal["base_rate"] == pytest.approx(sum(y) / len(y))
    assert cal["live"] is None


def test_decide_uses_the_calibrated_threshold(compressed):
    _, _, rows = compressed
    m = PreferenceModel(40, recall_target=0.9)
    assert m.train(rows)
    outcomes = set()
    for _, vec in rows[:40]:
        decision, p = m.decide(vec)
        assert decision == ("like" if p >= m.threshold else "dislike")
        outcomes.add(decision)
    # A fixed 0.5 cutoff would call a different set; this pins that both outcomes occur so the
    # equality above is not vacuous.
    assert outcomes == {"like", "dislike"}


def test_expected_false_pass_rate_is_the_liked_share_below_the_threshold(compressed):
    y, X, rows = compressed
    oof = _oof(y, X)
    pos = [s for s, liked in zip(oof, y, strict=True) if liked]
    base = thr.recall_threshold(pos, 0.9)
    below = [liked for s, liked in zip(oof, y, strict=True) if s < base]
    assert below and sum(below) > 0  # precondition: there are false passes to measure
    m = PreferenceModel(40, recall_target=0.9)
    m.train(rows)
    assert m.calibration["expected_false_pass_rate"] == pytest.approx(
        sum(below) / len(below)
    )


def _with_review_positives(compressed):
    """120 manual labels + 40 review 'would have liked' rows whose features look like passes.

    Hard positives: if they leaked into the calibration set they would drag the threshold down.
    """
    y, X, rows = compressed
    ry, rX = _dataset(40, seed=2, signal=0.05, like_rate=0.0)
    assert not any(ry)  # all negative-looking
    review = [(True, x.tolist(), "auto_review_random") for x in rX]
    manual = [(liked, vec, "") for liked, vec in rows]
    all_y = list(y) + [True] * 40
    all_X = np.concatenate([X, rX], axis=0)
    return manual + review, all_y, all_X, len(y)


def test_review_labels_train_the_classifier_but_are_not_calibration_labels(compressed):
    samples, all_y, all_X, n_manual = _with_review_positives(compressed)
    oof = _oof(all_y, all_X)
    manual_pos = [
        s for s, liked in zip(oof[:n_manual], all_y[:n_manual], strict=True) if liked
    ]
    all_pos = [s for s, liked in zip(oof, all_y, strict=True) if liked]
    want = thr.recall_threshold(manual_pos, 0.9)
    wrong = thr.recall_threshold(all_pos, 0.9)
    # Precondition: counting review rows as calibration labels WOULD have changed the answer.
    assert abs(want - wrong) > 1e-4

    m = PreferenceModel(40, recall_target=0.9)
    assert m.train(samples) is True
    assert m.threshold == pytest.approx(want, rel=1e-9)
    cal = m.calibration
    assert cal["calibration_labels"] == n_manual and cal["review_labels"] == 40
    # base_rate is over MANUAL labels only (review rows are all biased toward 'pass' candidates).
    assert cal["base_rate"] == pytest.approx(sum(all_y[:n_manual]) / n_manual)
    assert m.n_labels == n_manual + 40


def test_not_ready_without_enough_manual_likes_even_when_reviews_supply_them(
    compressed,
):
    y, X, rows = compressed
    # 3 manual likes (< min_per_class 5) + 30 review likes: the class counts are fine for a
    # legacy model, but there is nothing honest to calibrate on.
    manual_likes = [r for r in rows if r[0]][:3]
    manual_passes = [r for r in rows if not r[0]][:60]
    review = [
        (True, vec, "auto_review_near_cut")
        for _, vec in [r for r in rows if r[0]][3:23]
    ]
    samples = [(liked, vec, "") for liked, vec in manual_likes + manual_passes] + review

    assert PreferenceModel(40).train(samples) is True  # a legacy model happily trains
    m = PreferenceModel(40, recall_target=0.9)
    assert m.train(samples) is False
    assert not m.ready and m.calibration is None
    assert "manual liked" in m.not_ready_reason and "(have 3)" in m.not_ready_reason
    with pytest.raises(RuntimeError, match="not ready"):
        m.decide(rows[0][1])


def test_not_ready_when_out_of_fold_scoring_is_impossible():
    # One identity: every row shares the same face vector, so identity grouping yields a single
    # group and no grouped fold exists.  A legacy model trains on it; a recall-target model must
    # say so and must NOT quietly keep the fixed threshold.
    rng = np.random.default_rng(8)
    face = rng.normal(size=_FACE)
    face /= np.linalg.norm(face)
    rows = []
    for i in range(40):
        tail = rng.normal(size=_DIM - _FACE) / math.sqrt(_DIM - _FACE)
        rows.append((i % 2 == 0, np.concatenate([face, tail]).tolist()))
    assert PreferenceModel(10).train(rows) is True
    m = PreferenceModel(10, threshold=0.5, recall_target=0.9)
    assert m.train(rows) is False
    assert not m.ready and m.calibration is None
    assert "out-of-fold" in m.not_ready_reason
    assert m.threshold == 0.5  # untouched placeholder, and nothing is allowed to use it
    with pytest.raises(RuntimeError, match="not ready"):
        m.predict_proba(rows[0][1])


def test_zero_norm_face_vector_is_reported_not_crashed():
    rows = [(i % 2 == 0, [0.0] * 4 + [float(i)]) for i in range(30)]
    m = PreferenceModel(10, recall_target=0.9)
    assert m.train(rows) is False
    assert "identity grouping" in m.not_ready_reason


def test_recall_target_without_sklearn_is_not_ready_not_purepy(monkeypatch, compressed):
    _, _, rows = compressed
    monkeypatch.setattr(
        model_mod,
        "new_classifier",
        lambda: (_ for _ in ()).throw(ImportError("no sklearn")),
    )
    legacy = PreferenceModel(40)
    assert (
        legacy.train(rows) is True and legacy._impl == "purepy"
    )  # fallback still exists
    m = PreferenceModel(40, recall_target=0.9)
    assert m.train(rows) is False
    assert not m.ready and "scikit-learn" in m.not_ready_reason


def test_failed_retrain_drops_the_calibration_with_the_classifier(compressed):
    _, _, rows = compressed
    m = PreferenceModel(40, recall_target=0.9)
    assert m.train(rows)
    assert m.calibration is not None and m.max_threshold_lift > 0
    assert m.train(rows[:5]) is False
    assert not m.ready and m.calibration is None
    assert m.max_threshold_lift == 0.0  # no recall accounting, no contextual demotion
    assert "min_labels" in m.not_ready_reason


def test_a_recall_model_that_was_never_trained_allows_no_contextual_lift():
    assert PreferenceModel(10, recall_target=0.9).max_threshold_lift == 0.0
    assert PreferenceModel(10).max_threshold_lift == 0.045


def test_calibration_property_is_a_defensive_copy(compressed):
    _, _, rows = compressed
    m = PreferenceModel(40, recall_target=0.9)
    m.train(rows)
    snap = m.calibration
    snap["threshold"] = 0.123
    snap["live"] = {"x": 1}
    assert m.calibration["threshold"] == m.threshold != 0.123
    assert m.calibration["live"] is None


def test_max_threshold_lift_costs_at_most_the_budget(compressed):
    y, X, rows = compressed
    oof = _oof(y, X)
    pos = [s for s, liked in zip(oof, y, strict=True) if liked]
    lifts = {}
    for budget in (0.0, 0.02, 0.2):
        m = PreferenceModel(40, recall_target=0.9, contextual_recall_budget=budget)
        assert m.train(rows)
        want = thr.recall_threshold(
            pos, max(1e-6, 0.9 - budget)
        ) - thr.recall_threshold(pos, 0.9)
        want = max(0.0, min(0.045, want))
        assert m.max_threshold_lift == pytest.approx(want, abs=1e-12)
        assert 0.0 <= m.max_threshold_lift <= 0.045
        assert m.calibration["max_threshold_lift"] == m.max_threshold_lift
        lifts[budget] = m.max_threshold_lift
    assert lifts[0.0] == 0.0
    assert 0.0 < lifts[0.02] < lifts[0.2] <= 0.045
    # The lifted bar is exactly the one worth `budget` of recall, i.e. a valid policy input.
    policy = interaction.AutoSessionPolicy(
        base_threshold=m.threshold, max_threshold_lift=m.max_threshold_lift
    )
    assert policy.max_threshold_lift == m.max_threshold_lift


def test_live_review_raises_the_effective_target_and_lowers_the_threshold(compressed):
    _, _, rows = compressed
    base = PreferenceModel(40, recall_target=0.9)
    assert base.train(rows)
    f_exp = base.calibration["expected_false_pass_rate"]

    live = _live(
        random_reviewed=40, random_would_like=30, auto_likes=40, auto_passes=60
    )
    m = PreferenceModel(40, recall_target=0.9)
    assert m.train(rows, live_review=live)
    cal = m.calibration
    assert (
        thr.wilson_lower(30, 40) > f_exp
    )  # the gate opens: the bound beats the expectation
    extra = (30 / 40 - f_exp) * 0.6 / cal["base_rate"]
    assert extra > 0
    assert cal["live"]["extra"] == pytest.approx(extra)
    assert cal["effective_target"] == pytest.approx(min(0.99, 0.9 + extra))
    assert cal["recall_target"] == 0.9
    assert m.threshold < base.threshold
    # f_exp is read at the BASE threshold: if it followed the corrected (lower) threshold each
    # correction would shrink its own bar and feed back into itself.
    assert cal["expected_false_pass_rate"] == pytest.approx(f_exp)
    assert cal["live"]["observed_false_pass_lower"] == pytest.approx(
        thr.wilson_lower(30, 40)
    )


def test_live_review_below_ten_random_answers_changes_nothing(compressed):
    _, _, rows = compressed
    base = PreferenceModel(40, recall_target=0.9)
    base.train(rows)
    m = PreferenceModel(40, recall_target=0.9)
    assert m.train(rows, live_review=_live(random_reviewed=9, random_would_like=9))
    assert m.threshold == base.threshold
    live = m.calibration["live"]
    assert live["extra"] == 0.0 and live["random_reviewed"] == 9


def test_live_review_can_never_push_the_target_past_the_cap(compressed):
    _, _, rows = compressed
    m = PreferenceModel(40, recall_target=0.95)
    assert m.train(
        rows,
        live_review=_live(
            random_reviewed=200, random_would_like=200, auto_likes=0, auto_passes=500
        ),
    )
    assert m.calibration["effective_target"] == 0.99
    assert 0.0 < m.threshold < 1.0


def test_malformed_live_review_is_a_loud_error_not_a_silent_correction(compressed):
    _, _, rows = compressed
    m = PreferenceModel(40, recall_target=0.9)
    with pytest.raises(ValueError, match="random_would_like"):
        m.train(rows, live_review={**_live(), "random_would_like": 99})
    assert not m.ready


def test_decide_reads_score_and_cutoff_in_one_critical_section():
    """train() publishes a new classifier and threshold together under the model lock; decide()
    must not combine the OLD classifier's score with the NEW cutoff.  Simulate the publish
    landing between decide()'s score and its comparison: with one critical section the publish
    is blocked until decide() returns, so the decision uses the cutoff its score was made under."""
    m = PreferenceModel(10, threshold=0.5, min_per_class=1)
    assert m.train([(True, [2.0, 2.0])] * 6 + [(False, [-2.0, -2.0])] * 6)
    vec = [2.0, 2.0]
    assert (
        0.5 < m.predict_proba(vec) < 0.9999
    )  # precondition: the new cutoff WOULD flip it

    def publish():
        with m._lock:
            m.threshold = 0.9999

    publisher = threading.Thread(target=publish)
    real = m.predict_proba

    def racing_predict(v):
        p = real(v)
        publisher.start()
        publisher.join(
            timeout=0.3
        )  # one critical section: still blocked on the lock here
        return p

    m.predict_proba = racing_predict
    try:
        decision, _ = m.decide(vec)
    finally:
        publisher.join(timeout=15)
    assert decision == "like"  # decided under the cutoff in force when it was scored
    assert m.threshold == 0.9999  # and the publish did land afterwards


# --------------------------------------------------------------------------------------------
# PreferenceModel -- the scorer that is calibrated is the scorer that is deployed
# --------------------------------------------------------------------------------------------


def test_a_recall_model_deploys_the_fold_ensemble_it_calibrated(compressed):
    from sklearn.linear_model import LogisticRegression

    _, _, rows = compressed
    m = PreferenceModel(40, recall_target=0.9)
    assert m.train(rows)
    assert isinstance(m._clf, thr.FoldEnsemble)
    assert len(m._clf.models) == thr.CALIBRATION_FOLDS == 10
    # The deployed score is the plain mean of the fold models' scores.
    probe = rows[3][1]
    members = [mdl.predict_proba([probe])[0][1] for mdl in m._clf.models]
    assert m.predict_proba(probe) == pytest.approx(
        sum(members) / len(members), abs=1e-12
    )
    # A legacy fixed-threshold model keeps its single full-data fit, untouched.
    legacy = PreferenceModel(40)
    assert legacy.train(rows)
    assert isinstance(legacy._clf, LogisticRegression)


def _production_world(seed, n, test_n, strength=0.9, base=-0.3, k=12):
    """Learnable production-shaped data: unit-norm random faces, a CLIP-like tail whose latent
    factors drive the label, and a fresh stream from the same world to judge recall on."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(k, _DIM - _FACE))
    direction = rng.normal(size=k)
    direction /= np.linalg.norm(direction)

    def draw(count):
        face = rng.normal(size=(count, _FACE))
        face /= np.linalg.norm(face, axis=1, keepdims=True)
        latent = rng.normal(size=(count, k))
        tail = latent @ loadings + rng.normal(size=(count, _DIM - _FACE)) * math.sqrt(k)
        tail /= np.linalg.norm(tail, axis=1, keepdims=True)
        liked = rng.random(count) < 1.0 / (
            1.0 + np.exp(-(strength * latent @ direction + base))
        )
        return np.hstack([face, tail]), liked

    return draw(n), draw(test_n)


def test_the_cutoff_holds_its_recall_on_fresh_profiles_not_just_in_sample():
    """The cutoff was read from out-of-fold scores of models fit on 80% of the labels and then
    applied to ONE model fit on 100%.  Strong L2 makes more data mean larger weights, so the
    deployed scores were ~13-17% more spread out than the scores the cutoff came from and the
    cutoff sat too high: 0.93-0.94 held-out recall for a 0.95 target, on most trainings.
    Pinned two ways, on 12 seeded owners judged against 3000 fresh profiles each: the spread
    of the deployed positives' scores against the spread of the out-of-fold scores the cutoff
    was read from (measured 1.13 before, 0.97 now), and the mean held-out recall (0.942 before,
    0.963 now).  Training on all labels and deploying THAT model fails the first by a wide
    margin; an in-sample quantile fails the second."""
    ratios, recalls = [], []
    for seed in range(12):
        (X, y), (Xt, yt) = _production_world(seed, 240, 3000)
        m = PreferenceModel(40, recall_target=0.95)
        assert m.train(_pairs([bool(v) for v in y], X))
        deployed = m._clf.predict_proba(Xt)[:, 1]
        oof = thr.oof_scores(X, y.astype(int))
        ratios.append(deployed[yt].std() / oof[y].std())
        recalls.append(float((deployed[yt] >= m.threshold).mean()))
    assert 0.85 <= float(np.mean(ratios)) <= 1.05, np.mean(ratios)
    assert float(np.mean(recalls)) >= 0.95, np.mean(recalls)


def _raw_fold_oof(X, y, order):
    """OOF scores from sklearn's unshuffled StratifiedGroupKFold with rows in ``order``.

    The identity groups are recomputed on the reordered rows, exactly as the model does: DBSCAN
    numbers its clusters in the order it meets them, and StratifiedGroupKFold assigns folds from
    those ids, which is how row order reaches the cutoff at all.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    from operation_love.ranker.model import new_classifier

    Xo, yo = X[order], np.asarray(y)[order]
    go = identity_groups(Xo[:, :_FACE].tolist())
    scores = np.empty(len(order))
    for train, valid in StratifiedGroupKFold(n_splits=10).split(Xo, yo, groups=go):
        scores[valid] = (
            new_classifier().fit(Xo[train], yo[train]).predict_proba(Xo[valid])[:, 1]
        )
    out = np.empty(len(order))
    out[order] = scores
    return out


def test_the_calibration_is_a_function_of_the_label_set_not_of_row_order(compressed):
    """The same labels in a different order must give the same cutoff.  BigQuery's
    load_label_rows has no ORDER BY and an unshuffled StratifiedGroupKFold assigns folds from
    arrival order, so row order alone moved the cutoff (SD ~0.002-0.003 on the real labels)."""
    y, X, rows = compressed
    n = len(y)
    identity = list(range(n))
    shuffled = list(np.random.default_rng(5).permutation(n))
    # Precondition: WITHOUT the canonical ordering the order matters (fold assignment differs).
    a = _raw_fold_oof(X, [int(v) for v in y], identity)
    b = _raw_fold_oof(X, [int(v) for v in y], shuffled)
    assert np.abs(a - b).max() > 1e-4

    reference = PreferenceModel(40, recall_target=0.9)
    assert reference.train(rows)
    for seed in range(4):
        perm = np.random.default_rng(seed).permutation(n)
        m = PreferenceModel(40, recall_target=0.9)
        assert m.train([rows[i] for i in perm])
        assert m.threshold == reference.threshold
        assert (
            m.calibration["expected_false_pass_rate"]
            == reference.calibration["expected_false_pass_rate"]
        )
        probe = rows[0][1]
        assert m.predict_proba(probe) == pytest.approx(
            reference.predict_proba(probe), abs=1e-12
        )


def _rows_with_counts(likes, passes, seed=21):
    y, X = _dataset(600, seed=seed, signal=0.05, like_rate=0.5)
    liked = [i for i, v in enumerate(y) if v][:likes]
    passed = [i for i, v in enumerate(y) if not v][:passes]
    assert len(liked) == likes and len(passed) == passes
    return [(y[i], X[i].tolist(), "") for i in liked + passed]


def test_a_target_the_manual_likes_cannot_measure_is_not_ready_not_a_guess():
    # 12 manual likes clear min_per_class (5), but a 0.95 recall target cannot be measured
    # from 12 positives: the cutoff degenerates to the lowest liked score and the true recall
    # is a coin flip around 0.83 (measured 0.832 at n=5, 0.909 at n=10).  It used to train
    # "successfully" and report 0.950.
    rows = _rows_with_counts(12, 60)
    m = PreferenceModel(40, recall_target=0.95)
    assert m.train(rows) is False
    assert not m.ready and m.calibration is None
    assert "cannot be measured" in m.not_ready_reason
    assert (
        "12 manual liked" in m.not_ready_reason and "at least 19" in m.not_ready_reason
    )
    # The same labels measure a 0.9 target fine (needs 9), and 19 likes measure 0.95.
    assert PreferenceModel(40, recall_target=0.9).train(rows) is True
    assert (
        PreferenceModel(40, recall_target=0.95).train(_rows_with_counts(19, 60)) is True
    )
    assert (
        PreferenceModel(40, recall_target=0.95).train(_rows_with_counts(18, 60))
        is False
    )
    # A legacy model has no target to measure and is unaffected.
    assert PreferenceModel(40).train(rows) is True


def test_a_configured_target_above_the_live_cap_is_honoured():
    rows = _rows_with_counts(210, 210)
    m = PreferenceModel(40, recall_target=0.995)
    assert m.train(rows) is True, m.not_ready_reason
    cal = m.calibration
    assert cal["recall_target"] == cal["effective_target"] == 0.995
    capped = PreferenceModel(40, recall_target=0.99)
    assert capped.train(rows) is True
    assert capped.calibration["effective_target"] == 0.99
    # The 0.995 cutoff keeps more people than the 0.99 one; the old unconditional cap made the
    # two identical.
    assert m.threshold < capped.threshold
    # And the reported false-pass rate describes the cutoff actually in force.
    pos = [
        s
        for s, (liked, _, _) in zip(
            thr.oof_scores(
                np.array([r[1] for r in rows]),
                [int(r[0]) for r in rows],
                identity_groups(np.array([r[1] for r in rows])[:, :_FACE].tolist()),
            ),
            rows,
            strict=True,
        )
        if liked
    ]
    assert m.threshold == pytest.approx(thr.recall_threshold(pos, 0.995), rel=1e-9)


# --------------------------------------------------------------------------------------------
# evaluate(recall_target=...) -- the model-quality card must not grade a ranker on the labels the
# cutoff was fitted to
# --------------------------------------------------------------------------------------------


def _eval_samples(seed, strength, n=300):
    (X, y), _ = _production_world(seed, n, 1, strength=strength)
    return _pairs([bool(v) for v in y], X), X, y


def test_scoring_the_labels_the_cutoff_came_from_cannot_tell_good_from_useless():
    """The vacuous check this replaces.  The cutoff is read off out-of-fold scores of these same
    labels, so scoring them at that cutoff lands on the target (give or take sampling noise)
    whatever the ranker is: a ranker with NO signal and one with strong signal are
    indistinguishable by recall, and a recall check on either was green or red by chance alone,
    never because of model quality (before the model's folds changed it was green by arithmetic
    for every ranker).  Pinned as the precondition of the tests below, which show the nested
    evaluation DOES tell them apart."""
    from operation_love.ranker.evaluate import evaluate

    recalls = {}
    for strength in (0.0, 1.5):
        recalls[strength] = []
        for seed in (1, 2, 3):
            samples, X, y = _eval_samples(seed, strength=strength)
            cutoff = thr.recall_threshold(
                [s for s, v in zip(_oof(list(y), X), y, strict=True) if v], 0.95
            )
            in_sample = evaluate(samples, like_threshold=cutoff)
            assert in_sample["status"] == "ok"
            recalls[strength].append(in_sample["accepted_recall"])
    for strength, values in recalls.items():
        assert all(abs(v - 0.95) < 0.05 for v in values), (strength, values)
    useless, strong = (float(np.mean(recalls[s])) for s in (0.0, 1.5))
    assert abs(useless - strong) < 0.03  # recall does not separate them


def test_nested_evaluation_does_not_pass_a_ranker_with_no_signal():
    from operation_love.ranker.evaluate import evaluate

    for seed in (1, 2, 3):
        samples, _, _ = _eval_samples(seed, strength=0.0)
        r = evaluate(samples, recall_target=0.95)
        assert r["status"] == "ok" and r["calibration"] == "nested"
        assert 0.4 < r["roc_auc"][0] < 0.62  # precondition: genuinely no signal
        # At the 95%-recall cutoff it passes about the same share of dislikes as of likes:
        # nothing gained, and the "better than chance" check says so.
        assert r["discriminates"] is False, (seed, r["dislikes_passed_lower"])


def test_nested_evaluation_passes_a_ranker_that_has_signal():
    from operation_love.ranker.evaluate import evaluate

    for seed in (1, 2, 3):
        samples, _, _ = _eval_samples(seed, strength=1.5)
        r = evaluate(samples, recall_target=0.95)
        assert r["status"] == "ok" and r["roc_auc"][0] > 0.7
        assert r["discriminates"] is True, (seed, r["dislikes_passed_lower"])
        assert r["dislikes_passed_lower"] > r["chance_pass_rate"] >= 0.05 - 1e-12
        # Calibrated on training people only, so the held-out recall is an honest number near
        # the target (not pinned to it, which is the point), and its bounds bracket it.
        assert 0.85 < r["accepted_recall"] < 1.0
        assert r["accepted_recall"] <= r["accepted_recall_upper"] <= 1.0


def test_nested_evaluation_calibrates_each_fold_on_its_training_people_only(
    monkeypatch,
):
    """Mutating evaluate() back to one cutoff from ALL the positives (or from the held-out
    fold's own scores) makes the sizes below the total, not the training share."""
    from sklearn.model_selection import StratifiedGroupKFold

    import operation_love.ranker.evaluate as eval_mod

    samples, X, y = _eval_samples(2, strength=1.5)
    yv = np.asarray([int(v) for v in y])
    groups = identity_groups(X[:, :_FACE].tolist())
    expected = sorted(
        int(yv[train].sum())
        for train, _ in StratifiedGroupKFold(n_splits=5).split(X, yv, groups=groups)
    )
    seen = []
    real = eval_mod.recall_threshold
    monkeypatch.setattr(
        eval_mod,
        "recall_threshold",
        lambda pos, target: seen.append(len(pos)) or real(pos, target),
    )

    r = eval_mod.evaluate(samples, recall_target=0.95)

    assert r["status"] == "ok" and r["folds"] == 5
    assert sorted(seen) == expected
    assert max(seen) < int(yv.sum())  # never the whole label set
    assert len(r["fold_thresholds"]) == 5
    assert len(set(r["fold_thresholds"])) > 1  # a cutoff per fold, not one shared value


def test_nested_evaluation_reports_the_numbers_the_card_shows():
    from operation_love.ranker.evaluate import evaluate

    samples, _, _ = _eval_samples(3, strength=1.5)
    r = evaluate(samples, recall_target=0.9, like_threshold=0.4321)
    c = r["confusion"]
    assert (
        r["like_threshold"] == 0.4321 and r["recall_target"] == 0.9
    )  # echoed, not used
    assert r["accepted_recall"] == pytest.approx(
        c["true_accepted"] / (c["true_accepted"] + c["false_dislikes"])
    )
    assert r["dislikes_passed"] == pytest.approx(
        c["true_dislikes"] / (c["true_dislikes"] + c["false_likes"])
    )
    assert r["chance_pass_rate"] >= 1.0 - 0.9 - 1e-12
    assert r["dislikes_passed_lower"] < r["dislikes_passed"]
    assert sum(c.values()) == len(samples)


@pytest.mark.parametrize(
    "bad", [0, 0.0, 1, 1.0, -0.5, 2, True, "0.9", math.nan, math.inf]
)
def test_nested_evaluation_rejects_a_bad_target_as_a_status_not_a_raise(bad):
    from operation_love.ranker.evaluate import evaluate

    samples, _, _ = _eval_samples(1, strength=1.5, n=40)
    r = evaluate(samples, recall_target=bad)
    assert r["status"] == "error" and "recall_target" in r["message"]


def test_a_fixed_threshold_evaluation_is_unchanged_and_still_reports_the_extra_figures():
    from operation_love.ranker.evaluate import evaluate

    samples, _, _ = _eval_samples(1, strength=1.5, n=80)
    r = evaluate(samples)
    assert r["status"] == "ok" and "calibration" not in r and "fold_thresholds" not in r
    assert r["message"] == f"identity-grouped {r['folds']}-fold CV"
    assert r["like_threshold"] == 0.5


# --------------------------------------------------------------------------------------------
# fetch_* / format_calibration
# --------------------------------------------------------------------------------------------


class _RowsStore:
    def __init__(self, rows=None, pairs=None, stats=None, stats_error=None):
        self._rows, self._pairs = rows, pairs
        self._stats, self._stats_error = stats, stats_error
        self.stats_calls = []

    def load_label_rows(self):
        return self._rows

    def load_labels(self):
        return self._pairs

    def auto_review_stats(self, **kwargs):
        self.stats_calls.append(kwargs)
        if self._stats_error:
            raise self._stats_error
        return self._stats


def test_fetch_label_rows_prefers_rows_and_falls_back_to_pairs():
    both = _RowsStore(rows=[(True, [1.0], "")], pairs=[(True, [1.0])])
    assert thr.fetch_label_rows(both) == [(True, [1.0], "")]

    class PairsOnly:
        def load_labels(self):
            return [(False, [2.0])]

    assert thr.fetch_label_rows(PairsOnly()) == [(False, [2.0])]


def test_fetch_live_review_is_best_effort_and_loud():
    logs = []
    ok = _RowsStore(stats=_live())
    assert thr.fetch_live_review(ok, log=logs.append) == _live()
    assert ok.stats_calls == [{"app": None}]  # pooled across apps, like the labels
    assert logs == []

    class NoStats:
        pass

    assert thr.fetch_live_review(NoStats(), log=logs.append) is None
    assert logs == []  # a store without review support is not an error

    boom = _RowsStore(stats_error=RuntimeError("bq down"))
    assert thr.fetch_live_review(boom, log=logs.append) is None
    assert len(logs) == 1 and "RuntimeError" in logs[0] and "bq down" in logs[0]

    garbage = _RowsStore(stats={"random_reviewed": 5})
    assert thr.fetch_live_review(garbage, log=logs.append) is None
    assert len(logs) == 2 and "ValueError" in logs[1]


def test_fetch_training_inputs_pairs_rows_with_live_stats():
    store = _RowsStore(rows=[(True, [1.0], "")], stats=_live())
    rows, live = thr.fetch_training_inputs(store, log=lambda _m: None)
    assert rows == [(True, [1.0], "")] and live == _live()


def test_format_calibration_covers_legacy_unready_and_calibrated(compressed):
    _, _, rows = compressed
    assert "fixed threshold" in thr.format_calibration(PreferenceModel(10))

    unready = PreferenceModel(40, recall_target=0.9)
    assert "not trained yet" in thr.format_calibration(unready)
    unready.train(rows[:3])
    text = thr.format_calibration(unready)
    assert "NOT READY" in text and "0.900" in text and "min_labels" in text

    m = PreferenceModel(40, recall_target=0.9)
    m.train(rows, live_review=_live())
    text = thr.format_calibration(m)
    assert text.startswith("Calibration: recall target 0.900 -> effective ")
    assert f"threshold {m.threshold:.4f}" in text
    assert "30/40 random passes" in text and "max contextual lift" in text
    m.train(rows)
    assert "no live review data" in thr.format_calibration(m)


# --------------------------------------------------------------------------------------------
# RankerDecider.retrain
# --------------------------------------------------------------------------------------------


class _SpyModel:
    def __init__(self, recall_target=0.9):
        self.recall_target = recall_target
        self.calls = []

    def train(self, rows, **kwargs):
        self.calls.append((rows, kwargs))
        return True


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _decider(model, clock=None):
    return RankerDecider(None, None, model, **({"clock": clock} if clock else {}))


def test_retrain_passes_rows_with_sources_and_live_stats():
    store = _RowsStore(rows=[(True, [1.0], "auto_review_random")], stats=_live())
    model = _SpyModel()
    assert _decider(model).retrain(store) is True
    assert model.calls == [
        ([(True, [1.0], "auto_review_random")], {"live_review": _live()})
    ]


def test_retrain_reuses_live_stats_within_the_ttl_only():
    store = _RowsStore(rows=[], stats=_live())
    clock, model = _Clock(), _SpyModel()
    d = _decider(model, clock)
    d.retrain(store)
    clock.now += d.LIVE_REVIEW_TTL_S - 1
    d.retrain(store)
    assert len(store.stats_calls) == 1  # label count moved, stats did not: one query
    clock.now += 2
    d.retrain(store)
    assert len(store.stats_calls) == 2
    # A different store never inherits another store's stats.
    other = _RowsStore(rows=[], stats=_live(random_reviewed=11, random_would_like=3))
    d.retrain(other)
    assert len(other.stats_calls) == 1
    assert model.calls[-1][1]["live_review"]["random_reviewed"] == 11


def test_retrain_does_not_cache_a_failed_stats_read(capsys):
    store = _RowsStore(rows=[], stats_error=RuntimeError("bq down"))
    model = _SpyModel()
    d = _decider(model, _Clock())
    d.retrain(store)
    d.retrain(store)
    assert len(store.stats_calls) == 2  # asked again, not pinned to None for a minute
    assert all(kwargs == {"live_review": None} for _, kwargs in model.calls)
    assert capsys.readouterr().out.count("bq down") == 2


def test_retrain_skips_the_stats_query_for_a_fixed_threshold_model():
    store = _RowsStore(rows=[(True, [1.0], "")], stats=_live())
    model = _SpyModel(recall_target=None)
    _decider(model).retrain(store)
    assert store.stats_calls == []
    assert model.calls == [([(True, [1.0], "")], {})]  # and no live_review kwarg at all


def test_retrain_with_a_pairs_only_store_and_a_real_model():
    class PairsOnly:
        def load_labels(self):
            return [(True, [2.0, 2.0])] * 6 + [(False, [-2.0, -2.0])] * 6

    model = PreferenceModel(10, min_per_class=5)
    assert RankerDecider(None, None, model).retrain(PairsOnly()) is True


# --------------------------------------------------------------------------------------------
# supervisor wiring
# --------------------------------------------------------------------------------------------

_SUP_CONFIG = """
enabled_apps: [hinge]
mode: auto
apps:
  hinge: {{}}
paths:
  data_dir: {data_dir}
storage:
  backend: sqlite
opener:
  enabled: false
budget:
  run_budget_usd: 5.0
  pricing: {{}}
ranker:
  min_labels_to_engage: {min_labels}
  min_per_class: 5
{extra}
"""


class _Caps:
    @classmethod
    def detect(cls, android_adb_path=None):
        return cls()

    def banner(self):
        return ""

    def missing(self, *names):
        return []


class _FastEmbedder:
    def warmup(self):
        pass


class _FastQuality:
    def __init__(self, *a, **k):
        pass

    def warmup(self):
        pass


class _IdleDriver(DatingAppDriver):
    def open_session(self):
        pass

    def next_profile(self):
        return None

    def out_of_profiles(self):
        return True

    def like(self, opener=None, item_index=None, *, model_item_index=None):
        pass

    def dislike(self):
        pass

    def close(self):
        pass


class _SupStore:
    def __init__(self, rows=None, pairs=None, stats=None, stats_error=None):
        self._rows, self._pairs = rows, pairs
        self._stats, self._stats_error = stats, stats_error
        self.stats_calls = []
        self.closed = False

    def load_label_rows(self):
        return self._rows

    def load_labels(self):
        return self._pairs if self._pairs is not None else [r[:2] for r in self._rows]

    def auto_review_stats(self, **kwargs):
        self.stats_calls.append(kwargs)
        if self._stats_error:
            raise self._stats_error
        return self._stats

    def count_today(self, app):
        return 0

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def saved_summary(self):
        return ""


class _PairsOnlyStore:
    """The pre-existing duck type: no load_label_rows, no auto_review_stats."""

    def load_labels(self):
        return []

    def count_today(self, app):
        return 0

    def flush(self):
        pass

    def close(self):
        pass

    def saved_summary(self):
        return ""


def _run_supervisor(monkeypatch, tmp_path, store, *, min_labels=40, extra=""):
    spy = SimpleNamespace(models=[], workers=[])

    class SpyModel(PreferenceModel):
        def __init__(self, *a, **kw):
            spy.models.append({"args": a, "kwargs": kw, "trained": []})
            super().__init__(*a, **kw)

        def train(self, samples, *, live_review=None):
            spy.models[-1]["trained"].append((list(samples), live_review))
            return super().train(samples, live_review=live_review)

    real_worker = sup.Worker

    class SpyWorker(real_worker):
        def __init__(self, *a, **kw):
            spy.workers.append(kw)
            super().__init__(*a, **kw)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        _SUP_CONFIG.format(
            data_dir=str(tmp_path / "data"), min_labels=min_labels, extra=extra
        )
    )
    original_reason = sup.platforms.unavailable_reason
    monkeypatch.setattr(sup, "_ANDROID_LOCK_ROOT", tmp_path / "locks")
    monkeypatch.setattr(
        sup.cfg_mod, "_validate_hinge_auto_release_evidence", lambda _cfg: None
    )
    monkeypatch.setattr(
        sup.platforms,
        "unavailable_reason",
        lambda app, mode=None: (
            None if app == "hinge" and mode == "auto" else original_reason(app, mode)
        ),
    )
    monkeypatch.setattr(sup, "PreferenceModel", SpyModel)
    monkeypatch.setattr(sup, "Worker", SpyWorker)
    monkeypatch.setattr(sup, "Capabilities", _Caps)
    monkeypatch.setattr(sup, "Embedder", _FastEmbedder)
    monkeypatch.setattr(sup, "QualityFilter", _FastQuality)
    monkeypatch.setattr(sup, "make_store", lambda cfg: store)
    monkeypatch.setattr(sup, "make_driver", lambda app, cfg: _IdleDriver())
    monkeypatch.setattr(sup, "_install_signal_handlers", lambda stop: None)
    monkeypatch.setattr(sup, "_android_adb_preflight", lambda app, cfg: None)
    status = {}
    sup.run(
        str(cfg_path),
        on_status=lambda s: status.__setitem__("s", s),
        stop_event=threading.Event(),
    )
    return spy, status["s"].snapshot()


def test_supervisor_builds_the_model_from_config_and_trains_on_rows_plus_live_stats(
    monkeypatch, tmp_path, capsys
):
    rows = [(True, [1.0, 2.0], "auto_review_random"), (False, [3.0, 4.0], "")]
    store = _SupStore(rows=rows, stats=_live())
    spy, snap = _run_supervisor(
        monkeypatch,
        tmp_path,
        store,
        extra="  like_recall_target: 0.9\n  contextual_recall_budget: 0.05",
    )
    kwargs = spy.models[0]["kwargs"]
    assert kwargs["recall_target"] == 0.9
    assert kwargs["contextual_recall_budget"] == 0.05
    # The model saw the SOURCE-carrying rows (not the stripped pairs) and the live stats.
    assert spy.models[0]["trained"] == [(rows, _live())]
    assert store.stats_calls == [{"app": None}]
    assert snap["phase"] == "stopped"
    out = capsys.readouterr().out
    assert "Ranker: labels=2 ready=False" in out
    assert "uncalibrated (recall target 0.9)" in out
    # The calibration line sits directly under the Ranker line.
    lines = out.splitlines()
    ranker = next(i for i, ln in enumerate(lines) if ln.startswith("Ranker: labels="))
    assert lines[ranker + 1].startswith("Calibration: NOT READY at recall target 0.900")


def test_supervisor_prints_the_calibrated_cutoff(monkeypatch, tmp_path, capsys):
    y, X = _dataset(120, seed=1, signal=0.05)
    rows = [(liked, x.tolist(), "") for liked, x in zip(y, X, strict=True)]
    store = _SupStore(rows=rows, stats=_live())
    spy, _ = _run_supervisor(
        monkeypatch, tmp_path, store, extra="  like_recall_target: 0.9"
    )
    out = capsys.readouterr().out
    assert "Ranker: labels=120 ready=True" in out
    assert "recall target 0.9)" in out and "uncalibrated" not in out
    assert "Calibration: recall target 0.900 -> effective" in out
    assert "30/40 random passes" in out


def test_supervisor_review_stats_failure_is_loud_but_never_blocks_the_run(
    monkeypatch, tmp_path, capsys
):
    store = _SupStore(rows=[], stats_error=RuntimeError("stats table missing"))
    spy, snap = _run_supervisor(
        monkeypatch, tmp_path, store, extra="  like_recall_target: 0.9"
    )
    assert snap["phase"] == "stopped"
    assert spy.models[0]["trained"] == [([], None)]
    assert "stats table missing" in capsys.readouterr().out


def test_supervisor_still_runs_against_a_pairs_only_store(monkeypatch, tmp_path):
    spy, snap = _run_supervisor(monkeypatch, tmp_path, _PairsOnlyStore())
    assert snap["phase"] == "stopped"
    assert spy.models[0]["trained"] == [([], None)]


def test_supervisor_legacy_config_output_is_unchanged(monkeypatch, tmp_path, capsys):
    store = _PairsOnlyStore()
    spy, _ = _run_supervisor(monkeypatch, tmp_path, store)
    kwargs = spy.models[0]["kwargs"]
    assert (
        kwargs["recall_target"] is None and kwargs["contextual_recall_budget"] == 0.02
    )
    out = capsys.readouterr().out
    assert "Ranker: labels=0 ready=False (min=40, threshold=0.5)" in out
    assert "Calibration:" not in out


def test_supervisor_passes_the_review_share_to_every_worker(monkeypatch, tmp_path):
    spy, _ = _run_supervisor(
        monkeypatch,
        tmp_path,
        _PairsOnlyStore(),
        extra="auto_review:\n  enabled: true\n  random_share: 0.3",
    )
    assert [w["auto_review_random_share"] for w in spy.workers] == [0.3]


def test_supervisor_disables_pass_archiving_when_auto_review_is_off(
    monkeypatch, tmp_path
):
    spy, _ = _run_supervisor(
        monkeypatch,
        tmp_path,
        _PairsOnlyStore(),
        extra="auto_review:\n  enabled: false",
    )
    assert [w["auto_review_random_share"] for w in spy.workers] == [None]


def test_supervisor_passes_the_archive_retention_to_every_worker(monkeypatch, tmp_path):
    spy, _ = _run_supervisor(
        monkeypatch,
        tmp_path,
        _PairsOnlyStore(),
        extra="auto_review:\n  max_archived_passes: 123\n  max_age_days: 9",
    )
    assert [
        (w["auto_review_max_passes"], w["auto_review_max_age_days"])
        for w in spy.workers
    ] == [(123, 9)]


def test_supervisor_retention_defaults_to_the_shipped_bounds(monkeypatch, tmp_path):
    spy, _ = _run_supervisor(monkeypatch, tmp_path, _PairsOnlyStore())
    assert [
        (w["auto_review_max_passes"], w["auto_review_max_age_days"])
        for w in spy.workers
    ] == [(400, 30.0)]


def test_supervisor_retention_still_applies_when_archiving_is_switched_off(
    monkeypatch, tmp_path
):
    # `enabled: false` stops NEW passes being kept; the ones already stored are still strangers'
    # photos and must still age out.
    spy, _ = _run_supervisor(
        monkeypatch,
        tmp_path,
        _PairsOnlyStore(),
        extra="auto_review:\n  enabled: false",
    )
    assert [w["auto_review_random_share"] for w in spy.workers] == [None]
    assert [
        (w["auto_review_max_passes"], w["auto_review_max_age_days"])
        for w in spy.workers
    ] == [(400, 30.0)]


def test_auto_review_retention_helper():
    section = SimpleNamespace(max_archived_passes=50, max_age_days=2.5)
    assert sup._auto_review_retention(SimpleNamespace(auto_review=section)) == (50, 2.5)
    section = SimpleNamespace(max_archived_passes=0, max_age_days=0)
    assert sup._auto_review_retention(SimpleNamespace(auto_review=section)) == (
        None,
        None,
    )
    assert sup._auto_review_retention(SimpleNamespace()) == (None, None)


def test_auto_review_random_share_helper():
    cfg = SimpleNamespace(auto_review=SimpleNamespace(enabled=True, random_share=0.4))
    assert sup._auto_review_random_share(cfg) == 0.4
    cfg.auto_review.enabled = False
    assert sup._auto_review_random_share(cfg) is None
    assert sup._auto_review_random_share(SimpleNamespace()) is None


def test_ranker_threshold_text_never_shows_a_placeholder_as_a_cutoff(compressed):
    _, _, rows = compressed
    legacy_cfg = SimpleNamespace(
        ranker=SimpleNamespace(like_recall_target=None, like_threshold=0.5)
    )
    assert sup._ranker_threshold_text(legacy_cfg, PreferenceModel(10)) == "0.5"
    cfg = SimpleNamespace(
        ranker=SimpleNamespace(like_recall_target=0.9, like_threshold=0.5)
    )
    untrained = PreferenceModel(10, recall_target=0.9)
    assert "uncalibrated" in sup._ranker_threshold_text(cfg, untrained)
    m = PreferenceModel(40, recall_target=0.9)
    m.train(rows)
    assert sup._ranker_threshold_text(cfg, m).startswith(f"{m.threshold:.4f}")


# --------------------------------------------------------------------------------------------
# `python -m operation_love stats`
# --------------------------------------------------------------------------------------------


def _stats_config(tmp_path, ranker):
    import yaml

    cfg = {
        "enabled_apps": ["bumble"],
        "mode": "auto",
        "storage": {"backend": "sqlite"},
        "opener": {"enabled": False},
        "paths": {"data_dir": str(tmp_path), "db_file": str(tmp_path / "store.db")},
        "ranker": ranker,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_stats_show_prints_the_calibration_for_a_recall_target_config(tmp_path, capsys):
    from operation_love import stats
    from operation_love.ranker.store import SQLiteStore

    y, X = _dataset(120, seed=1, signal=0.05)
    store = SQLiteStore(tmp_path / "store.db")
    for liked, x in zip(y, X, strict=True):
        store.add_label("r1", "bumble", liked, x.tolist(), source="manual")
    for i in range(6):  # answers to auto-pass reviews: train, but never calibrate
        store.add_label(
            "r1", "bumble", i % 2 == 0, X[i].tolist(), source="auto_review_random"
        )
    store.close()

    path = _stats_config(
        tmp_path,
        {"min_labels_to_engage": 40, "like_recall_target": 0.9, "min_per_class": 5},
    )
    stats.show(str(path))
    out = capsys.readouterr().out
    assert "Labels  : 126" in out
    assert "Ranker  : ready=True" in out
    assert "recall target 0.900" in out
    assert "120 manual + 6 reviewed labels" in out
    # The calibrated cutoff is shown, not the 0.5 placeholder.
    cutoff = next(ln for ln in out.splitlines() if ln.startswith("Ranker  :")).split(
        "threshold "
    )[1]
    assert not cutoff.startswith("0.5")
    assert "Reviews : random 0/0 would-have-liked" in out


def test_stats_show_says_why_a_recall_model_is_not_ready_instead_of_asking_for_labels(
    tmp_path, capsys
):
    from operation_love import stats
    from operation_love.ranker.store import SQLiteStore

    store = SQLiteStore(tmp_path / "store.db")
    for i in range(12):  # plenty of labels, but every one is an auto-review answer
        store.add_label(
            "r1", "bumble", i % 2 == 0, [float(i), 1.0], source="auto_review_random"
        )
    store.close()
    path = _stats_config(
        tmp_path,
        {"min_labels_to_engage": 10, "like_recall_target": 0.9, "min_per_class": 5},
    )
    stats.show(str(path))
    out = capsys.readouterr().out
    assert "Ranker  : ready=False" in out and "uncalibrated" in out
    assert "Not ready: recall-target calibration needs at least 5 manual liked" in out
    assert "Seed ~" not in out  # 12 >= 10 labels: more labels is not the missing thing


def test_stats_show_legacy_config_output_is_unchanged(tmp_path, capsys):
    from operation_love import stats

    path = _stats_config(tmp_path, {"min_labels_to_engage": 40})
    stats.show(str(path))
    out = capsys.readouterr().out
    assert "Ranker  : ready=False  (min 40, threshold 0.5)" in out
    assert "Calibration" not in out
    assert "Seed ~40 more decisions in training mode" in out
