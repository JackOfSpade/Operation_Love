"""Personal preference model — logistic regression on YOUR swipe embeddings.

The "learns your taste" core. Trained on (liked, feature_vector) pairs collected
from your own swipes. Uses scikit-learn when available (fast, regularized); falls
back to a small pure-Python logistic regression so the interface works without
heavy deps (the fallback is fine for tests / low dimensions, slow for the real
~1280-dim vectors — install the `ml` extra for sklearn).

The feature dimension comes from the training rows, never from a constant: 1280 for
``ranker.feature_set: face_crop_v1``, 2048 for ``face_crop_whole_photo_v1`` (the v1 vector
followed by the 768-d whole-photo vector).  Either way the first 512 dims are the ArcFace
identity template that out-of-fold identity grouping reads.  A loader that cannot supply the
configured feature set (``threshold.FeatureSetIncomplete``) is published with
``mark_not_ready(exc.reason)`` -- the model is NOT READY with that reason, never trained on a
partial or zero-filled label set.

Cold-start: stays "not ready" until at least `min_labels` labels exist AND both
like/dislike classes are present, so the bot doesn't swipe blind on no signal.

Two decision-threshold regimes (see ranker/threshold.py for the full rationale):

* ``recall_target=None`` (legacy): the fixed ``threshold`` argument is the like cutoff.
* ``recall_target=0.95`` (e.g.): every ``train`` computes identity-grouped out-of-fold scores
  over ALL labels, reads the MANUAL labels' liked scores, and sets ``threshold`` to the highest
  cutoff that keeps that share of them above it.  The classifier it DEPLOYS is the average of
  the fold models that produced those scores (not a separate fit on all the labels), so the
  cutoff is applied to the same kind of scorer it was read from.  If the calibration cannot be
  computed the model is NOT ready (``not_ready_reason`` says why) -- it never quietly keeps the
  fixed threshold.
"""

from __future__ import annotations

import copy
import math
import threading
from numbers import Real

from .threshold import (
    effective_target as _effective_target,
    fit_folds,
    is_calibration_source,
    live_adjustment,
    min_positives_for_target,
    recall_threshold,
)

# Shared with ranker/evaluate.py's offline CV, so the reported accuracy can't
# silently drift from the classifier actually shipped in PreferenceModel.
SKLEARN_LOGREG_KWARGS = {"C": 0.1, "class_weight": "balanced", "max_iter": 1000}

# Mirrors interaction._THRESHOLD_MAX_LIFT (the most AutoSessionPolicy may lift a marginal
# like's bar).  Duplicated, not imported: interaction imports ranker.decider, which imports
# this package, so importing it here would be a cycle.  tests/test_recall_threshold.py pins the
# two constants equal.
DEFAULT_MAX_THRESHOLD_LIFT = 0.045
# The contextual-demotion budget is a recall cost in [0, MAX]: 0.2 is already "AUTO may pass a
# fifth more of the people you like"; anything larger is a typo, not a policy.
MAX_CONTEXTUAL_RECALL_BUDGET = 0.2


def new_classifier():
    """Strong L2 (small C) for ~1280-2048-d features on tens-hundreds of labels, and
    balanced class weights since likes/passes are usually imbalanced. Raises
    ImportError if scikit-learn isn't installed."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**SKLEARN_LOGREG_KWARGS)


def _sigmoid(z: float) -> float:
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _finite_real_as_float(value: object) -> float | None:
    """Return a finite real as ``float`` without leaking conversion errors.

    Some valid ``numbers.Real`` implementations -- including an arbitrarily large Python
    integer -- raise ``OverflowError`` when converted to float.  Public validation paths should
    reject those values with their documented ``ValueError`` rather than exposing that backend
    conversion detail to callers.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


class _PurePyLogReg:
    """Tiny batch gradient-descent logistic regression (no numpy/sklearn)."""

    def __init__(self, l2: float = 1.0, lr: float = 0.5, epochs: int = 300):
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.w: list[float] = []
        self.b: float = 0.0

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        n, d = len(X), len(X[0])
        self.w = [0.0] * d
        self.b = 0.0
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for xi, yi in zip(X, y, strict=True):
                p = _sigmoid(
                    self.b + sum(wj * xj for wj, xj in zip(self.w, xi, strict=True))
                )
                err = p - yi
                for j in range(d):
                    gw[j] += err * xi[j]
                gb += err
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j] / n)
            self.b -= self.lr * (gb / n)

    def predict_proba(self, x: list[float]) -> float:
        return _sigmoid(self.b + sum(wj * xj for wj, xj in zip(self.w, x, strict=True)))


class _CalibrationImpossible(Exception):
    """Recall-target calibration cannot be computed from these labels (reason in args[0])."""


class PreferenceModel:
    # min_labels has no default: it comes from config.yaml's ranker.min_labels_to_engage
    # (operation_love.config.RankerCfg) at every real call site, so a stale duplicate
    # default here could silently drift from what's actually configured. threshold's
    # default (0.5) is the standard binary-classifier decision boundary, and min_per_class's
    # default (5) is a reasonable floor, so both are fine to keep independently.
    #
    # recall_target / contextual_recall_budget have the same relationship to config.yaml's
    # ranker.like_recall_target / ranker.contextual_recall_budget.  recall_target=None keeps
    # the legacy fixed-threshold behaviour byte-for-byte; the budget's 0.02 is only consulted
    # when a recall target is set.
    def __init__(
        self,
        min_labels: int,
        threshold: float = 0.5,
        min_per_class: int = 5,
        recall_target: float | None = None,
        contextual_recall_budget: float = 0.02,
    ):
        if type(min_labels) is not int or min_labels <= 0:
            raise ValueError("min_labels must be a positive integer")
        if type(min_per_class) is not int or min_per_class <= 0:
            raise ValueError("min_per_class must be a positive integer")
        threshold_value = _finite_real_as_float(threshold)
        if threshold_value is None or not 0.0 < threshold_value < 1.0:
            raise ValueError("threshold must be a finite number in (0, 1)")
        target_value: float | None = None
        if recall_target is not None:
            target_value = _finite_real_as_float(recall_target)
            if target_value is None or not 0.0 < target_value < 1.0:
                raise ValueError(
                    "recall_target must be None or a finite number strictly between 0 and 1"
                )
        budget_value = _finite_real_as_float(contextual_recall_budget)
        if (
            budget_value is None
            or not 0.0 <= budget_value <= MAX_CONTEXTUAL_RECALL_BUDGET
        ):
            raise ValueError(
                "contextual_recall_budget must be a finite number in "
                f"[0, {MAX_CONTEXTUAL_RECALL_BUDGET}]"
            )
        self.min_labels = min_labels
        self.threshold = threshold_value
        self.min_per_class = min_per_class
        self.recall_target = target_value
        self.contextual_recall_budget = budget_value
        self.n_labels = 0
        # Why the model is not ready, for the operator (startup line, `stats`).  Pessimistic
        # until a train() completes: an exception mid-train leaves this string in place.
        self.not_ready_reason: str | None = "not trained yet"
        # What KIND of not-ready this is, when a caller said so (``mark_not_ready(kind=...)``;
        # ``threshold.FEATURE_SET_INCOMPLETE_KIND`` for missing whole-photo rows), else None --
        # a cold start or a failed train.  Lets the worker name the right remedy without
        # parsing the reason text.
        self.not_ready_kind: str | None = None
        self._clf = None
        self._impl = None
        self._feature_dim: int | None = None
        self._calibration: dict | None = None
        # supervisor shares one PreferenceModel through every Worker.  A retrain must be
        # observed as one state transition: readers either use the completed classifier or wait
        # for the completed failure/not-ready state, never see the temporary cleared sentinel
        # while fit() is running.
        self._lock = threading.RLock()

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._clf is not None

    @property
    def calibration(self) -> dict | None:
        """Snapshot of the recall calibration behind ``threshold``, or ``None``.

        ``None`` for a legacy fixed-threshold model and for a recall-target model that is not
        ready.  Keys: recall_target, effective_target, threshold, expected_false_pass_rate (at
        the BASE threshold, i.e. before the live correction -- computing it at the corrected
        threshold would let each correction lower its own bar and run away),
        base_rate, calibration_labels (manual), review_labels, live (dict or None) and
        max_threshold_lift.  A deep copy: callers cannot mutate model state through it.
        """
        with self._lock:
            return copy.deepcopy(self._calibration)

    @property
    def max_threshold_lift(self) -> float:
        """The most AutoSessionPolicy may lift a marginal like's bar.

        Legacy models: the policy's own 0.045 ceiling.  Recall-target models: the lift whose
        recall cost is at most ``contextual_recall_budget`` (capped at 0.045); ``0.0`` while
        no calibration exists, because a lift with no recall accounting is exactly what the
        recall target forbids.
        """
        with self._lock:
            if self.recall_target is None:
                return DEFAULT_MAX_THRESHOLD_LIFT
            if self._calibration is None:
                return 0.0
            return self._calibration["max_threshold_lift"]

    def _not_ready(self, reason: str) -> bool:
        self.not_ready_reason = reason
        return False

    def mark_not_ready(
        self, reason: str, *, n_labels: int | None = None, kind: str | None = None
    ) -> bool:
        """Publish NOT READY with ``reason`` when the labels could not even be loaded.

        For a loader failure that is a verdict rather than a bug -- ``FeatureSetIncomplete``
        (pass ``exc.reason``, ``n_labels=exc.labels`` and ``kind=FEATURE_SET_INCOMPLETE_KIND``).
        Like a failed ``train`` it replaces the state as a whole: a classifier from an earlier
        label set is never kept serving.  ``kind`` is published as ``not_ready_kind`` until the
        next ``train``.  Returns False (the new ``ready``).
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("mark_not_ready needs a non-empty reason")
        if n_labels is not None and (type(n_labels) is not int or n_labels < 0):
            raise ValueError("n_labels must be None or a non-negative integer")
        if kind is not None and (not isinstance(kind, str) or not kind.strip()):
            raise ValueError("kind must be None or a non-empty string")
        with self._lock:
            self._clf = None
            self._impl = None
            self._feature_dim = None
            self._calibration = None
            if n_labels is not None:
                self.n_labels = n_labels
            self.not_ready_kind = kind
            return self._not_ready(reason)

    def train(
        self,
        samples: list[tuple[bool, list[float]]]
        | list[tuple[bool, list[float], str | None]],
        *,
        live_review: dict | None = None,
    ) -> bool:
        """samples: list of (liked, feature_vector) pairs or (liked, feature_vector, source)
        triples. Returns whether the model is ready.

        ``source`` only matters with a ``recall_target``: labels whose source starts with
        ``auto_review`` train the classifier but are excluded from threshold calibration.
        ``live_review`` is ``store.auto_review_stats()`` output (see ranker/threshold.py); it is
        ignored without a recall target.
        """
        # Retraining is an all-or-nothing replacement.  In particular, do not keep serving a
        # classifier trained on an older label set when the current one is insufficient or its
        # fit fails: that would make ``retrain`` appear to have failed loudly while the swipe
        # loop still acts on stale preferences.  Clear the readiness sentinel before every
        # validation/fit attempt; a newly fitted classifier is published only after fit()
        # succeeds below.
        # Hold the same lock readers use for the complete fit. A separate "training" flag
        # would let ready() return an obsolete classifier while the new labels are being fitted;
        # clearing without this lock made the opposite mistake, exposing a transient not-ready
        # state. Blocking briefly is the only honest answer for a shared mutable model.
        with self._lock:
            self._clf = None
            self._impl = None
            self._feature_dim = None
            self._calibration = None
            self.not_ready_reason = "training did not complete"
            self.not_ready_kind = None
            if not isinstance(samples, (list, tuple)):
                raise ValueError(
                    "samples must be a list or tuple of (liked, feature_vector) pairs"
                )
            self.n_labels = len(samples)
            X, y, feature_dim, is_manual = self._validated_samples(samples)
            if self.n_labels < self.min_labels:
                return self._not_ready(
                    f"{self.n_labels} labels < min_labels {self.min_labels}"
                )
            if len(set(y)) < 2:  # need both like and dislike examples
                return self._not_ready("labels contain only one class")
            n_likes = sum(y)
            if min(n_likes, len(y) - n_likes) < self.min_per_class:
                return self._not_ready(
                    f"fewer than min_per_class {self.min_per_class} labels in one class "
                    f"({n_likes} liked / {len(y) - n_likes} passed)"
                )
            if self.recall_target is not None:
                manual_likes = sum(
                    1
                    for liked, manual in zip(y, is_manual, strict=True)
                    if liked and manual
                )
                if manual_likes < self.min_per_class:
                    return self._not_ready(
                        f"recall-target calibration needs at least {self.min_per_class} "
                        f"manual liked labels (have {manual_likes}); answers to auto-pass "
                        "reviews train the model but are not calibration labels"
                    )
                # A recall target is a statement about the people the owner would like, read
                # off their manual likes.  With fewer than this the target cannot be measured:
                # the cutoff degenerates to the lowest liked score, whose true recall is only
                # n/(n+1) (0.83 at 5 likes, 0.91 at 10, against a reported 0.95), so say so
                # instead of reporting a number the labels do not support.
                needed_likes = min_positives_for_target(self.recall_target)
                if manual_likes < needed_likes:
                    return self._not_ready(
                        f"recall target {self.recall_target:g} cannot be measured with "
                        f"{manual_likes} manual liked labels (needs at least "
                        f"{needed_likes}); label more people you like, or lower "
                        "ranker.like_recall_target"
                    )
            # The ONLY intended fallback is "sklearn isn't installed" (ImportError) -> the
            # pure-Python LR. Scope the try to the import alone: a genuine fit FAILURE (e.g.
            # NaN/inf in the feature vectors, which sklearn rejects) must surface loudly, not
            # be swallowed into a silently-degraded model that reports ready=True and then
            # auto-swipes on garbage (the pure-Python LR has no NaN guard and would fit NaN
            # weights). This is training/inference code, not best-effort logging.
            try:
                clf = new_classifier()
            except ImportError:
                if self.recall_target is not None:
                    # Out-of-fold scoring needs scikit-learn; the pure-Python fallback would
                    # leave a recall target that can never be calibrated.  Not ready, loudly.
                    return self._not_ready(
                        "scikit-learn is not installed; recall-target calibration needs "
                        "out-of-fold scores (install the `ml` extra)"
                    )
                clf = _PurePyLogReg()
                clf.fit(X, y)
                impl = "purepy"
            else:
                impl = "sklearn"
                # A recall-target model deploys the fold ensemble (below) instead, so it does
                # not spend a full-data fit it would then throw away.  Outside the except: a
                # real bug in .fit() (bad shapes, NaN/inf embeddings, a sklearn version
                # incompatibility) propagates instead of silently and permanently degrading to
                # the pure-Python fallback with no signal -- inside the fold fits too.
                if self.recall_target is None:
                    clf.fit(X, y)
            calibration = None
            if self.recall_target is not None:
                try:
                    calibration, clf = self._calibrate(X, y, is_manual, live_review)
                except _CalibrationImpossible as exc:
                    return self._not_ready(str(exc))
            # Publish classifier, threshold and calibration together, still under the lock.
            if calibration is not None:
                self.threshold = calibration["threshold"]
            self._calibration = calibration
            self._impl, self._clf = impl, clf
            self._feature_dim = feature_dim
            self.not_ready_reason = None
            return True

    def _calibrate(
        self,
        X: list[list[float]],
        y: list[int],
        is_manual: list[bool],
        live_review: dict | None,
    ) -> tuple[dict, object]:
        """Compute the recall-target threshold from identity-grouped OOF scores.

        Returns ``(calibration, scorer)``: the scorer is the fold ensemble whose members
        produced those out-of-fold scores, and is what the model deploys.

        Raises ``_CalibrationImpossible`` (with the operator-facing reason) when the labels
        cannot support it; any other exception is a real bug and propagates.
        """
        assert self.recall_target is not None
        try:
            import numpy as np

            from .evaluate import _FACE_DIMS, identity_groups

            Xa = np.asarray(X, dtype=float)
            try:
                groups = identity_groups(Xa[:, : min(_FACE_DIMS, Xa.shape[1])].tolist())
            except ValueError as exc:
                raise _CalibrationImpossible(
                    f"identity grouping for out-of-fold scoring failed: {exc}"
                ) from exc
            fit = fit_folds(Xa, y, groups, classifier_factory=new_classifier)
        except ImportError as exc:
            raise _CalibrationImpossible(
                f"recall-target calibration needs numpy and scikit-learn ({exc})"
            ) from exc
        if fit is None:
            raise _CalibrationImpossible(
                "out-of-fold scoring is impossible with these labels (need at least 2 "
                "identity-grouped folds whose training side has both classes; have "
                f"{len(set(groups))} identities, {sum(y)} liked / {len(y) - sum(y)} passed)"
            )
        manual = [
            (float(s), liked)
            for s, liked, m in zip(fit.oof, y, is_manual, strict=True)
            if m
        ]
        pos = [s for s, liked in manual if liked]
        # train() already required >= min_per_class manual likes, so `pos` is nonempty and
        # `manual` is too; base_rate is therefore well defined and > 0.
        base_rate = len(pos) / len(manual)
        base_threshold = recall_threshold(pos, self.recall_target)
        below = [liked for s, liked in manual if s < base_threshold]
        expected_false_pass = (sum(below) / len(below)) if below else 0.0
        live = live_adjustment(
            live_review,
            expected_false_pass_rate=expected_false_pass,
            base_rate=base_rate,
        )
        effective = _effective_target(
            self.recall_target, live["extra"] if live is not None else 0.0
        )
        threshold = recall_threshold(pos, effective)
        # Contextual demotion (AutoSessionPolicy) lifts the bar for a marginal like.  Every
        # lifted pass is recall spent, so allow only the lift whose cost is the budget: the
        # distance from the threshold at `effective` up to the threshold at `effective -
        # budget` (a LOWER recall target has a HIGHER cutoff).
        lower_target = max(1e-6, effective - self.contextual_recall_budget)
        lift = recall_threshold(pos, lower_target) - threshold
        lift = max(0.0, min(DEFAULT_MAX_THRESHOLD_LIFT, lift))
        review_labels = len(y) - len(manual)
        calibration = {
            "recall_target": self.recall_target,
            "effective_target": effective,
            "threshold": threshold,
            "expected_false_pass_rate": expected_false_pass,
            "base_rate": base_rate,
            "calibration_labels": len(manual),
            "review_labels": review_labels,
            "live": live,
            "max_threshold_lift": lift,
        }
        return calibration, fit.ensemble

    @staticmethod
    def _validated_samples(
        samples: list[tuple[bool, list[float]]] | tuple[tuple[bool, list[float]], ...],
    ) -> tuple[list[list[float]], list[int], int | None, list[bool]]:
        """Return classifier-ready copies, rejecting poisoned/ragged training data.

        sklearn catches most of these itself, but the supported pure-Python fallback does not:
        it can otherwise publish NaN weights as a ready model.  Validation belongs before the
        implementation choice so both backends have the same safety contract.

        The fourth element flags each row as a MANUAL calibration label.  A 2-tuple (or a
        triple whose source is None/"" or does not start with ``auto_review``) is manual.
        """
        X: list[list[float]] = []
        y: list[int] = []
        is_manual: list[bool] = []
        feature_dim: int | None = None
        for ordinal, sample in enumerate(samples):
            if not isinstance(sample, (list, tuple)) or len(sample) not in (2, 3):
                raise ValueError(
                    f"sample {ordinal} must be a (liked, feature_vector) pair or a "
                    "(liked, feature_vector, source) triple"
                )
            liked, vector = sample[0], sample[1]
            if (
                len(sample) == 3
                and sample[2] is not None
                and not isinstance(sample[2], str)
            ):
                raise ValueError(f"sample {ordinal} source must be a string or None")
            manual = True if len(sample) == 2 else is_calibration_source(sample[2])
            if type(liked) is not bool:
                raise ValueError(f"sample {ordinal} liked label must be exactly bool")
            if not isinstance(vector, (list, tuple)) or not vector:
                raise ValueError(
                    f"sample {ordinal} feature vector must be a nonempty sequence"
                )
            if feature_dim is None:
                feature_dim = len(vector)
            elif len(vector) != feature_dim:
                raise ValueError(
                    f"sample {ordinal} feature vector has dimension {len(vector)}; "
                    f"expected {feature_dim}"
                )
            row: list[float] = []
            for dimension, value in enumerate(vector):
                converted = _finite_real_as_float(value)
                if converted is None:
                    raise ValueError(
                        f"sample {ordinal} feature {dimension} must be a finite real number"
                    )
                row.append(converted)
            X.append(row)
            y.append(1 if liked else 0)
            is_manual.append(manual)
        return X, y, feature_dim, is_manual

    @staticmethod
    def _validated_prediction_vector(
        vec: list[float], expected_dim: int
    ) -> list[float]:
        if not isinstance(vec, (list, tuple)) or len(vec) != expected_dim:
            actual = len(vec) if isinstance(vec, (list, tuple)) else type(vec).__name__
            raise ValueError(
                f"prediction feature vector must have dimension {expected_dim} (got {actual})"
            )
        out: list[float] = []
        for dimension, value in enumerate(vec):
            converted = _finite_real_as_float(value)
            if converted is None:
                raise ValueError(
                    f"prediction feature {dimension} must be a finite real number"
                )
            out.append(converted)
        return out

    def predict_proba(self, vec: list[float]) -> float:
        with self._lock:
            if self._clf is None:
                raise RuntimeError("PreferenceModel is not ready (not enough labels)")
            if (
                self._feature_dim is None
            ):  # defensive invariant; a ready model always sets it
                raise RuntimeError(
                    "PreferenceModel is ready without a feature dimension"
                )
            validated = self._validated_prediction_vector(vec, self._feature_dim)
            if self._impl == "sklearn":
                raw_probability = self._clf.predict_proba([validated])[0][1]
            else:
                raw_probability = self._clf.predict_proba(validated)
            probability = _finite_real_as_float(raw_probability)
            if probability is None or not 0.0 <= probability <= 1.0:
                raise RuntimeError(
                    f"PreferenceModel returned invalid probability {raw_probability!r}"
                )
            return probability

    def decide(self, vec: list[float]) -> tuple[str, float]:
        # One critical section for the score AND the cutoff it is compared with: train()
        # publishes a new classifier and a new threshold together, and a decision must not mix
        # the old classifier's score with the new classifier's cutoff.
        with self._lock:
            p = self.predict_proba(vec)
            return ("like" if p >= self.threshold else "dislike", p)
