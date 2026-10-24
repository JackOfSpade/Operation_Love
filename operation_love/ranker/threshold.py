"""Recall-target calibration for the personal ranker.

Why this exists.  The owner would rather the bot auto-LIKE someone they would pass than
auto-PASS someone they would like.  The ranker's probabilities are compressed (identity-grouped
out-of-fold scores sit in a narrow band around 0.5, AUC ~0.70), so a fixed ``like_threshold:
0.5`` quietly passes a large share of the people the owner would have liked.  Instead of
choosing a probability cutoff, the owner states a RECALL: "the bot may pass at most 5% of the
people I would like" (``like_recall_target: 0.95``), and this module turns that into the
cutoff that holds it.

How it is measured.  The cutoff is read from OUT-OF-FOLD scores (every profile is scored by a
classifier that never saw it, with the same person never straddling a train/validation split),
on MANUAL labels only.  Four ideas keep that honest:

* **Calibration labels are manual labels.**  Answers the owner gives to archived AUTO passes
  (source ``auto_review...``) are training signal but are NOT a fair sample of "people the
  owner would like": every one of them is a profile the model had already decided to pass.
  Calibrating on them would measure the model against its own past mistakes.
* **The scorer that is calibrated is the scorer that is deployed.**  An earlier version read the
  cutoff from 5 fold models fit on 80% of the labels and applied it to ONE model fit on 100%.
  With strong L2 more data means larger weights and a wider score spread, so the cutoff sat
  too high for the deployed model and held-out recall landed 1.5-2.5 points under the target on
  most trainings.  Now the deployed scorer IS the average of the fold models whose out-of-fold
  scores set the cutoff (``FoldEnsemble``), so the cutoff is read off the same kind of scorer
  it is applied to.  Measured against 6000 fresh profiles it holds the target or lands slightly
  above it (the average of correlated models is a touch narrower than one model), which is the
  direction the owner prefers.
* **Finite-sample cutoff.**  The k-th highest of n exchangeable positive scores leaves a fresh
  positive above it with probability only ``k/(n+1)``, so the in-sample quantile
  ``ceil(target*n)`` systematically under-delivers (a 0.95 target at n=40 delivers ~0.93).
  ``recall_threshold`` uses ``ceil(target*(n+1))``, and a model with too few manual likes to
  measure the target at all (``min_positives_for_target``) is NOT ready instead of reporting a
  target it cannot meet.
* **Live correction.**  The random-pool reviews (a uniform sample of what AUTO really passed)
  measure the live false-pass rate.  When its one-sided 95% lower bound is above the rate the
  OOF scores predicted, the effective recall target is raised by the recall that measured gap
  costs.

Everything numeric is imported lazily (numpy/scikit-learn) so ``ranker.model`` keeps its
documented "works without heavy deps" import surface.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

# A label whose ``source`` starts with this prefix is an answer to an archived AUTO profile
# ("auto_review_random" / "auto_review_near_cut" / "auto_review_like"). Everything else -- including legacy rows
# whose source is NULL or "" -- is a manual training label and is eligible for calibration.
REVIEW_SOURCE_PREFIX = "auto_review"

# A computed threshold must be a usable probability bound for the swipe loop
# (AutoSessionPolicy rejects 0 and 1).  Clamping a cutoff DOWN to 0.999 can only admit more
# positives (recall stays >= target); clamping one UP to 0.001 could drop a positive scoring
# below 0.001, which no real classifier output at this regularisation produces.
MIN_THRESHOLD = 0.001
MAX_THRESHOLD = 0.999

# Never let a live correction push the target past this: a recall of 1.0 is "like everyone".
# It bounds the LIVE INCREMENT only (see ``effective_target``): an owner who configures a higher
# target themselves gets that target, not a silently relaxed one.
MAX_EFFECTIVE_TARGET = 0.99
# Folds behind the out-of-fold scores AND the deployed fold ensemble.  More folds put each fold
# model closer to the data size it stands in for; 10 trains on 90% for the same number of fits
# the old 5-fold-plus-one-full-fit scheme spent.
CALIBRATION_FOLDS = 10
# Live evidence below this many RANDOM-pool answers is noise; ignore it entirely.
MIN_RANDOM_REVIEWS = 10
# One-sided 95% (normal quantile) for the Wilson bound.
WILSON_Z = 1.645

# `ranker.feature_set`: which vector the ranker trains and scores on.  `face_crop_v1` is the
# 1280-d face-crop vector stored in `labels.embedding` (and the default: nothing else is read).
# `face_crop_whole_photo_v1` appends the profile's whole-photo CLIP vector from the
# `photo_features` table (vision/photo_cards.py), 1280 + 768 = 2048 dims; the first 512 dims stay
# the ArcFace identity template that out-of-fold grouping reads.
FEATURE_SET_FACE_CROP_V1 = "face_crop_v1"
FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1 = "face_crop_whole_photo_v1"
FEATURE_SETS = (FEATURE_SET_FACE_CROP_V1, FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1)
DEFAULT_FEATURE_SET = FEATURE_SET_FACE_CROP_V1
# What FeatureSetIncomplete tells the operator to run.
BACKFILL_PHOTO_FEATURES_COMMAND = "python -m tools.backfill_photo_features"
# `PreferenceModel.not_ready_kind` (and the AUTO worker's stop_kind) for a ranker left NOT READY
# by FeatureSetIncomplete, so no surface mistakes it for a cold start.
FEATURE_SET_INCOMPLETE_KIND = "feature_set_incomplete"


def is_calibration_source(source: object) -> bool:
    """True when a label with this ``source`` counts as a manual calibration label."""
    if source is None:
        return True
    if not isinstance(source, str):
        raise ValueError(f"label source must be a string or None (got {source!r})")
    return not source.startswith(REVIEW_SOURCE_PREFIX)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number (got {value!r})")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be a finite real number (got {value!r})")
    return converted


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer (got {value!r})")
    return int(value)


def _validated_recall_target(target: object) -> float:
    t = _finite(target, "recall target")
    if not 0.0 < t < 1.0:
        raise ValueError(
            f"recall target must be strictly between 0 and 1 (got {target!r})"
        )
    return t


def _needed_count(n: int, t: float) -> int:
    """How many of ``n`` calibration positives the cutoff must keep, finite-sample corrected.

    ``ceil(t * (n + 1))``, not ``ceil(t * n)``: for exchangeable scores the k-th highest of n
    leaves a FRESH positive above it with probability ``k / (n + 1)``, so the in-sample count
    delivers only ``ceil(t*n) / (n+1)`` -- below ``t`` for every n and far below it for small n
    (measured: a 0.95 target gives 0.909 at n=10, 0.906 at n=20, 0.927 at n=40, 0.941 at n=100).
    The result is clamped to ``[1, n]``; when ``t * (n+1) > n`` the target cannot be measured
    and the cutoff is the minimum score (``min_positives_for_target`` is where callers refuse
    that case up front).  The epsilon absorbs binary-float noise (0.07 * 100 ==
    7.000000000000001) that would otherwise round the count UP by one and silently buy extra,
    unrequested recall.
    """
    return min(n, max(1, math.ceil(t * (n + 1) - 1e-9)))


def min_positives_for_target(target: float) -> int:
    """The fewest calibration positives at which ``target`` recall can be measured at all.

    Below this ``_needed_count`` exceeds n, so ``recall_threshold`` can only return the lowest
    positive score and a reported "95% recall" is a number nothing supports: 19 for 0.95, 9 for
    0.90, 99 for 0.99.  ``PreferenceModel.train`` refuses to calibrate on fewer, and says so.
    """
    t = _validated_recall_target(target)
    return max(1, math.ceil(t / (1.0 - t) - 1e-9))


def recall_threshold(pos_scores: Sequence[float], target: float) -> float:
    """The HIGHEST cutoff ``t`` expected to keep ``target`` of FRESH positives above it.

    Sorted descending, that is the score at index ``ceil(target * (n + 1)) - 1`` (see
    ``_needed_count`` for why ``n + 1``): no higher cutoff keeps that many of the n scores
    given (ties included).  The result is clamped into ``[MIN_THRESHOLD, MAX_THRESHOLD]`` so it
    is always a valid probability bound.  ``target`` must be strictly inside (0, 1); an empty
    or non-finite score list is an error, not a default.
    """
    t = _validated_recall_target(target)
    scores = [_finite(s, "positive score") for s in pos_scores]
    if not scores:
        raise ValueError("recall_threshold needs at least one positive score")
    scores.sort(reverse=True)
    needed = _needed_count(len(scores), t)
    return min(MAX_THRESHOLD, max(MIN_THRESHOLD, scores[needed - 1]))


def wilson_lower(k: int, n: int, z: float = WILSON_Z) -> float:
    """One-sided lower confidence bound of a binomial proportion (Wilson score interval).

    ``z=1.645`` is the one-sided 95% quantile.  ``n == 0`` has no evidence and returns 0.0 --
    the conservative value for the caller, which only ever acts on a bound ABOVE an expectation.
    """
    k = _count(k, "k")
    n = _count(n, "n")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    z = _finite(z, "z")
    if z < 0:
        raise ValueError("z must be non-negative")
    if n == 0:
        return 0.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z2 / (4.0 * n * n)))
    return min(1.0, max(0.0, (centre - margin) / denom))


class FoldEnsemble:
    """The fold classifiers, averaged: the scorer a recall-target model DEPLOYS.

    Its members are exactly the models whose out-of-fold scores chose the cutoff, so the cutoff
    is read off the same kind of scorer it is applied to (see the module docstring for the
    full-fit mismatch this replaces).  Quacks like a fitted scikit-learn classifier for the one
    call the ranker makes: ``predict_proba(X) -> (n, 2)``.
    """

    def __init__(self, models: Sequence[object]):
        models = tuple(models)
        if not models:
            raise ValueError("a FoldEnsemble needs at least one fold model")
        self.models = models

    def predict_proba(self, X):
        import numpy as np

        total = None
        for model in self.models:
            proba = np.asarray(model.predict_proba(X), dtype=float)
            total = proba if total is None else total + proba
        return total / len(self.models)


@dataclass(frozen=True)
class FoldFit:
    """Out-of-fold scores for every row, plus the fold models that produced them."""

    oof: object  # numpy array, one P(like) per input row, in the caller's row order
    ensemble: FoldEnsemble


def _canonical_order(X, yv) -> list[int]:
    """Row indices sorted by a content hash, so fold assignment ignores row order.

    ``StratifiedGroupKFold`` (unshuffled) assigns folds from the order rows arrive in, and
    ``BigQueryStore.load_label_rows`` has no ORDER BY: the same labels in a different order
    gave cutoffs differing by an SD of ~0.002-0.003 on the real data, i.e. two trainings of
    identical labels disagreed about which people the bot passes.  Sorting by content makes the
    calibration a function of the label SET.  Identical rows (same vector, same label) are
    interchangeable, so ties need no further rule.
    """
    import hashlib

    import numpy as np

    keys = []
    for i in range(X.shape[0]):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(b"\x01" if yv[i] else b"\x00")
        digest.update(np.ascontiguousarray(X[i], dtype=np.float64).tobytes())
        keys.append(digest.digest())
    return sorted(range(X.shape[0]), key=keys.__getitem__)


def fit_folds(
    X,
    y,
    groups: Sequence[int] | None = None,
    n_splits: int = CALIBRATION_FOLDS,
    *,
    classifier_factory: Callable[[], object] | None = None,
) -> FoldFit | None:
    """Identity-grouped out-of-fold ``P(like)`` for every row AND the fold ensemble, or ``None``.

    Uses ``model.new_classifier()`` -- the SAME hyperparameters as the deployed ranker -- and
    ``StratifiedGroupKFold`` so a person's rows never straddle train/validation.  ``groups``
    defaults to ``evaluate.identity_groups`` over the first 512 dims (the L2-normalised ArcFace
    template), exactly what the offline evaluator uses.  The folds are assigned in a canonical,
    row-order-independent order (``_canonical_order``).

    Returns ``None`` (never a partial vector) when fewer than 2 folds are possible, when any
    fold's TRAINING side lacks a class (its validation rows could not be scored, and a
    threshold read off a subset of rows would silently calibrate on a biased sample), or when a
    score comes back non-finite.  Raises ``ImportError`` if scikit-learn is unavailable and
    ``ValueError`` for malformed input; the caller decides how to report those.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    if classifier_factory is None:
        from .model import (
            new_classifier as classifier_factory,
        )  # lazy: avoids an import cycle

    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    X = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=int)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("oof_scores needs a nonempty 2-D feature matrix")
    if yv.shape != (X.shape[0],) or not np.isin(yv, (0, 1)).all():
        raise ValueError("oof_scores needs one 0/1 label per feature row")
    if groups is None:
        from .evaluate import _FACE_DIMS, identity_groups

        groups = identity_groups(X[:, : min(_FACE_DIMS, X.shape[1])].tolist())
    if len(groups) != X.shape[0]:
        raise ValueError("oof_scores needs one group id per feature row")

    likes = int(yv.sum())
    passes = int(yv.shape[0]) - likes
    splits = min(int(n_splits), len(set(groups)), likes, passes)
    if splits < 2:
        return None

    order = _canonical_order(X, yv)
    relabel: dict[object, int] = {}
    ordered_groups = [relabel.setdefault(groups[i], len(relabel)) for i in order]
    Xo, yo = X[order], yv[order]

    scores_ordered = np.full(X.shape[0], np.nan)
    models = []
    for train_idx, valid_idx in StratifiedGroupKFold(n_splits=splits).split(
        Xo, yo, groups=ordered_groups
    ):
        if len(set(yo[train_idx].tolist())) < 2:
            return None  # this fold's rows cannot be scored out-of-fold
        clf = classifier_factory()
        clf.fit(Xo[train_idx], yo[train_idx])
        proba = np.asarray(clf.predict_proba(Xo[valid_idx]), dtype=float)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError("classifier predict_proba must return an (n, 2) array")
        scores_ordered[valid_idx] = proba[:, 1]
        models.append(clf)
    if len(models) < 2 or not np.isfinite(scores_ordered).all():
        return None
    if ((scores_ordered < 0.0) | (scores_ordered > 1.0)).any():
        raise ValueError("classifier produced a probability outside [0, 1]")
    scores = np.empty_like(scores_ordered)
    scores[order] = scores_ordered
    return FoldFit(oof=scores, ensemble=FoldEnsemble(models))


def oof_scores(
    X,
    y,
    groups: Sequence[int] | None = None,
    n_splits: int = CALIBRATION_FOLDS,
    *,
    classifier_factory: Callable[[], object] | None = None,
):
    """Just the out-of-fold scores of :func:`fit_folds` (``None`` when impossible)."""
    fit = fit_folds(X, y, groups, n_splits, classifier_factory=classifier_factory)
    return None if fit is None else fit.oof


def parse_live_review(live_review: Mapping | None) -> dict | None:
    """Validate the store's ``auto_review_stats`` dict; return the fields we use.

    ``None`` stays ``None`` (no review data at all).  A present-but-malformed mapping raises
    ``ValueError`` rather than being coerced: a live correction built from garbage would move
    the bot's recall in an unaccountable direction.

    ``auto_model_passes`` is optional (a store that predates it simply cannot tell the two
    apart): the AUTO passes that were genuine MODEL passes, i.e. the ones that entered the
    archive.  ``auto_passes`` also counts likes the session policy or the like-ratio ceiling
    demoted to a pass, which the random pool never samples; defaulting to ``auto_passes`` is
    the old, inflated behaviour and only applies to a store without the field.
    """
    if live_review is None:
        return None
    if not isinstance(live_review, Mapping):
        raise ValueError(
            f"live_review must be a mapping or None (got {type(live_review).__name__})"
        )
    out: dict[str, int] = {}
    for key in ("random_reviewed", "random_would_like", "auto_likes", "auto_passes"):
        if key not in live_review:
            raise ValueError(f"live_review is missing {key!r}")
        out[key] = _count(live_review[key], f"live_review[{key!r}]")
    if out["random_would_like"] > out["random_reviewed"]:
        raise ValueError(
            "live_review random_would_like cannot exceed random_reviewed "
            f"({out['random_would_like']} > {out['random_reviewed']})"
        )
    if live_review.get("auto_model_passes") is None:
        out["auto_model_passes"] = out["auto_passes"]
    else:
        out["auto_model_passes"] = _count(
            live_review["auto_model_passes"], "live_review['auto_model_passes']"
        )
        if out["auto_model_passes"] > out["auto_passes"]:
            raise ValueError(
                "live_review auto_model_passes cannot exceed auto_passes "
                f"({out['auto_model_passes']} > {out['auto_passes']})"
            )
    return out


def live_adjustment(
    live_review: Mapping | None,
    *,
    expected_false_pass_rate: float,
    base_rate: float,
) -> dict | None:
    """Evidence-based raise of the recall target, from the random-pool reviews.

    A test, then an estimate.  The TEST is whether ``wilson_lower(k, n)`` -- the one-sided 95%
    lower bound of the random-pool would-like share ``k/n`` -- is above ``f_exp``, the
    false-pass rate the OOF scores predicted.  Only when it is does the target move, and then by
    the POINT estimate of the gap::

        extra = (k/n - f_exp) * pass_share / base_rate

    where ``pass_share`` is the share of AUTO decisions that were MODEL passes (the ones the
    random pool samples) and ``base_rate`` the liked share of calibration labels.  Read it as:
    (excess liked-among-passed) x (passes per profile) / (likes per profile) = the extra
    fraction of all likes the bot is losing.

    Why not subtract the Wilson margin itself (the first version did): that estimates the
    shortfall at its 95% LOWER bound, so a real recall shortfall had to exceed ~0.02 before
    anything moved and the loop then recovered only about half of it (simulated: a 0.012 score
    shift took live recall to 0.924, not 0.95).  Using the bound only as the significance gate
    keeps the "never move on noise" protection and applies an unbiased correction once the
    evidence is real.  If anything it over-corrects slightly after passing the gate, which is
    the direction the owner prefers; ``effective_target`` still caps it.

    Returns ``None`` when there is no review data; otherwise the ``live`` dict with
    ``extra == 0.0`` until at least ``MIN_RANDOM_REVIEWS`` random answers exist.
    """
    parsed = parse_live_review(live_review)
    if parsed is None:
        return None
    k, n = parsed["random_would_like"], parsed["random_reviewed"]
    decisions = parsed["auto_likes"] + parsed["auto_passes"]
    pass_share = parsed["auto_model_passes"] / decisions if decisions else 0.0
    lower = wilson_lower(k, n)
    extra = 0.0
    if n >= MIN_RANDOM_REVIEWS and base_rate > 0.0 and lower > expected_false_pass_rate:
        extra = (k / n - expected_false_pass_rate) * pass_share / base_rate
    return {
        "random_reviewed": n,
        "random_would_like": k,
        "observed_false_pass_lower": lower,
        "pass_share": pass_share,
        "extra": extra,
    }


def effective_target(target: float, extra: float) -> float:
    """``max(target, min(0.99, target + extra))``.

    The live correction can never reach 'like everyone': it may RAISE the target up to 0.99.
    The cap bounds the increment only.  Applying it to the configured target as well (the first
    version) silently relaxed ``like_recall_target: 0.995`` to 0.99, so the 'effective' target
    sat below the owner's own and the reported false-pass rate described a cutoff the bot was
    not using.
    """
    return max(target, min(MAX_EFFECTIVE_TARGET, target + max(0.0, extra)))


def require_feature_set(value: object) -> str:
    """``value`` when it names a known ranker feature set; ``ValueError`` otherwise."""
    if not isinstance(value, str) or value not in FEATURE_SETS:
        raise ValueError(
            f"ranker feature set must be one of {list(FEATURE_SETS)} (got {value!r})"
        )
    return value


class FeatureSetIncomplete(Exception):
    """Labels lack the stored features the configured feature set needs.

    Raised by :func:`fetch_label_rows` in ``face_crop_whole_photo_v1`` mode when any label has
    no usable ``photo_features`` row: no row at all, or a stored zero-card row (written before
    zero photo cards became an extraction failure; the backfill re-lists and recomputes those).
    Such a label is never dropped or zero-filled (that would train on a different label set
    than the one the owner made, or on a fabricated feature); the ranker is NOT READY with
    :attr:`reason` until the backfill has filled them.  ``missing`` labels out of ``labels``
    lack a usable row, ``zero_cards`` of them because their row records zero photo cards.
    """

    def __init__(self, missing: int, labels: int, zero_cards: int = 0):
        self.missing = _count(missing, "missing")
        self.labels = _count(labels, "labels")
        self.zero_cards = _count(zero_cards, "zero_cards")
        if self.zero_cards > self.missing:
            raise ValueError("zero_cards must not exceed missing")
        super().__init__(self.missing, self.labels, self.zero_cards)

    @property
    def reason(self) -> str:
        """The operator-facing not-ready reason (also ``str(exc)``)."""
        zero = (
            f" ({self.zero_cards} of them stored with zero photo cards)"
            if self.zero_cards
            else ""
        )
        return (
            f"{self.missing} of {self.labels} labels lack whole-photo features{zero}; "
            f"run {BACKFILL_PHOTO_FEATURES_COMMAND}"
        )

    def __str__(self) -> str:
        return self.reason


def _whole_photo_contract() -> tuple[str, int]:
    """``(WHOLE_PHOTO_FEATURE_VERSION, WHOLE_PHOTO_DIMS)``, imported lazily.

    The vision layer owns the constants; importing it only when the whole-photo feature set is
    in use keeps the default path's import surface unchanged.
    """
    from ..vision.photo_cards import WHOLE_PHOTO_DIMS, WHOLE_PHOTO_FEATURE_VERSION

    return WHOLE_PHOTO_FEATURE_VERSION, WHOLE_PHOTO_DIMS


def _whole_photo_label_rows(store) -> list:
    """``(liked, face_crop_v1 ++ whole_photo, source)`` for every label, or a raise.

    A label with no row at all, or whose winning row records ZERO photo cards
    (``has_photo_row`` True, vector None), is counted, and any such label raises
    :class:`FeatureSetIncomplete`.  Zero photo cards is an extraction failure, never a feature:
    there is no zero block (``RankerDecider`` refuses to score one too).
    """
    loader = getattr(store, "load_label_feature_rows", None)
    if not callable(loader):
        raise ValueError(
            f"ranker feature set {FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1!r} needs a store "
            f"with load_label_feature_rows ({type(store).__name__} has none)"
        )
    version, dims = _whole_photo_contract()
    rows = loader(version)
    out = []
    missing = zero_cards = 0
    for ordinal, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 5:
            raise ValueError(
                f"label feature row {ordinal} must be a (liked, embedding, source, "
                "photo_vector, has_photo_row) 5-tuple"
            )
        liked, embedding, source, photo, has_row = row
        if type(has_row) is not bool:
            raise ValueError(
                f"label feature row {ordinal} has_photo_row must be a bool"
            )
        if not has_row:
            if photo is not None:
                raise ValueError(
                    f"label feature row {ordinal} has a photo vector but no photo row"
                )
            missing += 1
            continue
        if photo is None:
            missing += 1
            zero_cards += 1
            continue
        block = list(photo)
        if len(block) != dims:
            raise ValueError(
                f"label feature row {ordinal} photo vector has dimension "
                f"{len(block)}; expected {dims}"
            )
        out.append((liked, [*embedding, *block], source))
    if missing:
        raise FeatureSetIncomplete(
            missing=missing, labels=len(rows), zero_cards=zero_cards
        )
    return out


def fetch_label_rows(store, feature_set: str = FEATURE_SET_FACE_CROP_V1) -> list:
    """The store's labels for ``PreferenceModel.train``, in the configured feature set.

    ``face_crop_v1`` (the default): ``load_label_rows()`` (``(liked, embedding, source)``
    triples) when the store has it, else ``load_labels()`` (``(liked, embedding)`` pairs --
    every pair is a manual calibration label).  Duck-typed on purpose: test doubles and older
    stores implement only part of the surface.  Nothing else is read.

    ``face_crop_whole_photo_v1``: the same labels in the same order, each embedding the v1
    vector followed by its ``WHOLE_PHOTO_DIMS`` whole-photo vector (see
    :func:`_whole_photo_label_rows`); raises :class:`FeatureSetIncomplete` when any label lacks
    a usable ``photo_features`` row.
    """
    if require_feature_set(feature_set) == FEATURE_SET_FACE_CROP_V1:
        loader = getattr(store, "load_label_rows", None)
        return loader() if callable(loader) else store.load_labels()
    return _whole_photo_label_rows(store)


def fetch_training_inputs(
    store,
    feature_set: str = FEATURE_SET_FACE_CROP_V1,
    *,
    log: Callable[[str], None] = print,
) -> tuple[list, dict | None]:
    """Everything ``PreferenceModel.train`` needs from a store: (rows, live_review).

    ``feature_set`` is :func:`fetch_label_rows`' (a :class:`FeatureSetIncomplete` propagates,
    before any review stats are read).  Live review stats are an ADD-ON: a store without
    ``auto_review_stats`` yields ``None`` silently, and an exception there is logged loudly and
    treated as ``None`` -- reviews must never block a run.
    """
    return fetch_label_rows(store, feature_set), fetch_live_review(store, log=log)


def fetch_live_review(store, *, log: Callable[[str], None] = print) -> dict | None:
    """Best-effort ``store.auto_review_stats``; see :func:`fetch_training_inputs`.

    Stats are pooled across apps (``app=None``) because the model and its label pool are shared
    across every enabled app; scoping the false-pass evidence to one app would compare it to a
    cross-app expectation.
    """
    stats = getattr(store, "auto_review_stats", None)
    if not callable(stats):
        return None
    try:
        live = stats(app=None)
        parse_live_review(
            live
        )  # reject garbage here, where "best-effort" can absorb it
        return live
    except Exception as exc:  # noqa: BLE001 - reviews are an add-on; never block a run
        log(
            "Auto-review stats unavailable "
            f"({type(exc).__name__}: {exc}); training WITHOUT the live false-pass correction."
        )
        return None


def format_calibration(model) -> str:
    """One human-readable line describing a model's recall calibration (for startup/stats).

    Accepts anything exposing ``calibration`` / ``not_ready_reason`` / ``recall_target``
    (a ``PreferenceModel``).  Legacy fixed-threshold models report that explicitly.
    """
    target = getattr(model, "recall_target", None)
    if target is None:
        return "Calibration: fixed threshold (ranker.like_recall_target is not set)"
    cal = getattr(model, "calibration", None)
    if cal is None:
        reason = getattr(model, "not_ready_reason", None) or "not trained yet"
        return f"Calibration: NOT READY at recall target {target:.3f} ({reason})"
    parts = [
        f"Calibration: recall target {cal['recall_target']:.3f}"
        f" -> effective {cal['effective_target']:.3f}"
        f", threshold {cal['threshold']:.4f}"
        f", expected false-pass {cal['expected_false_pass_rate']:.1%}"
        f", base rate {cal['base_rate']:.1%}"
        f", {cal['calibration_labels']} manual + {cal['review_labels']} reviewed labels"
    ]
    live = cal.get("live")
    if live is None:
        parts.append(", no live review data")
    else:
        parts.append(
            f", live {live['random_would_like']}/{live['random_reviewed']} random passes "
            f"would-have-liked (lower bound {live['observed_false_pass_lower']:.1%}, "
            f"target +{live['extra']:.3f})"
        )
    parts.append(f", max contextual lift {cal['max_threshold_lift']:.4f}")
    return "".join(parts)
