"""Personal preference model — logistic regression on YOUR swipe embeddings.

The "learns your taste" core. Trained on (liked, feature_vector) pairs collected
from your own swipes. Uses scikit-learn when available (fast, regularized); falls
back to a small pure-Python logistic regression so the interface works without
heavy deps (the fallback is fine for tests / low dimensions, slow for the real
~1280-dim vectors — install the `ml` extra for sklearn).

Cold-start: stays "not ready" until at least `min_labels` labels exist AND both
like/dislike classes are present, so the bot doesn't swipe blind on no signal.
"""
from __future__ import annotations

import math
import threading

# Shared with ranker/evaluate.py's offline CV, so the reported accuracy can't
# silently drift from the classifier actually shipped in PreferenceModel.
SKLEARN_LOGREG_KWARGS = {"C": 0.1, "class_weight": "balanced", "max_iter": 1000}


def new_classifier():
    """Strong L2 (small C) for ~1280-d features on tens-hundreds of labels, and
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
                p = _sigmoid(self.b + sum(wj * xj for wj, xj in zip(self.w, xi, strict=True)))
                err = p - yi
                for j in range(d):
                    gw[j] += err * xi[j]
                gb += err
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j] / n)
            self.b -= self.lr * (gb / n)

    def predict_proba(self, x: list[float]) -> float:
        return _sigmoid(self.b + sum(wj * xj for wj, xj in zip(self.w, x, strict=True)))


class PreferenceModel:
    # min_labels has no default: it comes from config.yaml's ranker.min_labels_to_engage
    # (operation_love.config.RankerCfg) at every real call site, so a stale duplicate
    # default here could silently drift from what's actually configured. threshold's
    # default (0.5) is the standard binary-classifier decision boundary, and min_per_class's
    # default (5) is a reasonable floor, so both are fine to keep independently.
    def __init__(self, min_labels: int, threshold: float = 0.5, min_per_class: int = 5):
        self.min_labels = min_labels
        self.threshold = threshold
        self.min_per_class = min_per_class
        self.n_labels = 0
        self._clf = None
        self._impl = None
        # supervisor shares one PreferenceModel through every Worker.  A retrain must be
        # observed as one state transition: readers either use the completed classifier or wait
        # for the completed failure/not-ready state, never see the temporary cleared sentinel
        # while fit() is running.
        self._lock = threading.RLock()

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._clf is not None

    def train(self, samples: list[tuple[bool, list[float]]]) -> bool:
        """samples: list of (liked, feature_vector). Returns whether the model is ready."""
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
            self.n_labels = len(samples)
            if self.n_labels < self.min_labels:
                return False
            X = [v for (_, v) in samples]
            y = [1 if liked else 0 for (liked, _) in samples]
            if len(set(y)) < 2:   # need both like and dislike examples
                return False
            n_likes = sum(y)
            if min(n_likes, len(y) - n_likes) < self.min_per_class:
                return False
            # The ONLY intended fallback is "sklearn isn't installed" (ImportError) -> the
            # pure-Python LR. Scope the try to the import alone: a genuine fit FAILURE (e.g.
            # NaN/inf in the feature vectors, which sklearn rejects) must surface loudly, not
            # be swallowed into a silently-degraded model that reports ready=True and then
            # auto-swipes on garbage (the pure-Python LR has no NaN guard and would fit NaN
            # weights). This is training/inference code, not best-effort logging.
            try:
                clf = new_classifier()
            except ImportError:
                clf = _PurePyLogReg()
                clf.fit(X, y)
                self._impl, self._clf = "purepy", clf
            else:
                # Outside the except: a real bug in .fit() (bad shapes, NaN/inf embeddings,
                # a sklearn version incompatibility) propagates instead of silently and
                # permanently degrading to the pure-Python fallback with no signal.
                clf.fit(X, y)
                self._impl, self._clf = "sklearn", clf
            return True

    def predict_proba(self, vec: list[float]) -> float:
        with self._lock:
            if self._clf is None:
                raise RuntimeError("PreferenceModel is not ready (not enough labels)")
            if self._impl == "sklearn":
                return float(self._clf.predict_proba([vec])[0][1])
            return self._clf.predict_proba(vec)

    def decide(self, vec: list[float]) -> tuple[str, float]:
        p = self.predict_proba(vec)
        return ("like" if p >= self.threshold else "dislike", p)
