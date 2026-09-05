"""Leakage-free, identity-grouped evaluation of the personal ranker.

Shared by the terminal CLI (tools/eval_aggregation.py) and the hub GUI card, so both
report the same numbers. Clusters labels by FACE identity — the first 512 dims of
each 1280-d vector are the L2-normalized ArcFace template — so the same person never
straddles a train/test split (that would let the model memorize a face and inflate
the score), then runs stratified K-fold CV of the LogisticRegression ranker.

`evaluate()` returns a JSON-able dict (status + counts + metrics) so it can be served
as-is over HTTP; `format_report()` renders it for the terminal.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

from .model import SKLEARN_LOGREG_KWARGS, new_classifier   # shared ranker hyperparameters (no drift)

_FACE_DIMS = 512   # first 512 of the 1280-d vector = L2-normed ArcFace identity template
_IDENTITY_EPS = 0.5   # DBSCAN cosine-distance threshold -> cosine similarity >= 0.5,
                      # the buffalo_l same-identity threshold (see identity_groups below)


def _validated_controls(n_splits: object, eps: object) -> tuple[int, float]:
    """Validate the two numerical controls before handing them to scikit-learn.

    Letting DBSCAN or StratifiedGroupKFold validate these made bad API input look like
    malformed stored data, and ``n_splits=True`` could quietly become a one-fold request.
    Cosine distance is bounded by two, so larger values add no useful behavior either.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise ValueError("eps must be a finite number in (0, 2]")
    try:
        normalized_eps = float(eps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("eps must be a finite number in (0, 2]") from exc
    if not math.isfinite(normalized_eps) or not 0.0 < normalized_eps <= 2.0:
        raise ValueError("eps must be a finite number in (0, 2]")
    return int(n_splits), normalized_eps


def identity_groups(face_vectors: list[list[float]], eps: float = _IDENTITY_EPS) -> list[int]:
    """Cluster rows by face identity via DBSCAN on cosine distance (eps=0.5 -> cosine
    similarity >= 0.5, the buffalo_l same-identity threshold). min_samples=1 so a face
    with no near neighbor becomes its own singleton group. One int id per input row."""
    # Keep the public control contract independent of data cardinality. Without this before
    # the empty fast path, ``identity_groups([], eps=0)`` silently accepted a configuration
    # that the same function correctly rejected once it received its first face vector.
    _, normalized_eps = _validated_controls(2, eps)
    if not face_vectors:
        return []
    import numpy as np
    from sklearn.cluster import DBSCAN
    vectors = np.asarray(face_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError("face vectors must be a nonempty rectangular matrix of finite numbers")
    # sklearn's cosine metric produces NaN for a zero vector. Reject it explicitly instead of
    # returning a backend-version-specific error (or, worse, an arbitrary identity grouping).
    if np.any(np.linalg.norm(vectors, axis=1) == 0.0):
        raise ValueError("face vectors must not contain zero-norm rows")
    return DBSCAN(eps=normalized_eps, min_samples=1, metric="cosine").fit_predict(vectors).tolist()


def evaluate(samples: list[tuple[bool, list[float]]], n_splits: int = 5,
             eps: float = _IDENTITY_EPS, like_threshold: float = 0.5) -> dict:
    """Identity-grouped, stratified K-fold CV. Returns a JSON-able dict: status,
    label counts, distinct identities, ranking metrics, and out-of-fold decisions at
    ``like_threshold``. Never raises — failure modes come back as a status + message."""
    empty_base = {
        "labels": 0, "likes": 0, "passes": 0, "identities": None,
        "folds": 0, "roc_auc": None, "pr_auc": None, "brier": None,
        "base_rate": 0.0, "like_threshold": None, "accepted_recall": None,
        "false_dislike_rate": None, "confusion": None,
    }
    try:
        n_splits, eps = _validated_controls(n_splits, eps)
        if isinstance(like_threshold, bool) or not isinstance(like_threshold, Real):
            raise ValueError("like_threshold must be a finite number in [0, 1]")
        threshold = float(like_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("like_threshold must be a finite number in [0, 1]")
    except (ValueError, TypeError, OverflowError) as exc:
        return {**empty_base, "status": "error",
                "message": f"evaluation controls are invalid: {exc}"}
    try:
        rows = list(samples)
        y: list[int] = []
        for ordinal, sample in enumerate(rows):
            if not isinstance(sample, (list, tuple)) or len(sample) != 2:
                raise ValueError(f"sample {ordinal} is not a (liked, embedding) pair")
            liked, _ = sample
            if type(liked) is not bool:
                raise ValueError(f"sample {ordinal} liked label is not exactly bool")
            y.append(1 if liked else 0)
    except Exception as exc:  # noqa: BLE001 - this API promises a status dict for all inputs
        return {**empty_base, "status": "error",
                "message": f"evaluation input is malformed: {type(exc).__name__}: {exc}"}

    n = len(rows)
    likes, passes = sum(y), n - sum(y)
    base = {
        "labels": n, "likes": likes, "passes": passes, "identities": None,
        "folds": 0, "roc_auc": None, "pr_auc": None, "brier": None,
        "base_rate": (likes / n) if n else 0.0, "like_threshold": threshold,
        "accepted_recall": None, "false_dislike_rate": None, "confusion": None,
    }
    if n < 10 or len(set(y)) < 2:
        return {**base, "status": "insufficient_data",
                "message": f"Need ≥10 labels with both like & pass "
                           f"(have {n}: {likes} like / {passes} pass)."}
    try:
        import numpy as np
        from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_auc_score
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": "no_sklearn", "message": f"scikit-learn unavailable: {exc}"}

    # Wrapped so this never raises (malformed/ragged embeddings, sklearn edge cases):
    # the GUI relies on a status dict, and the CLI calls evaluate() with no guard.
    try:
        X = np.asarray([emb for _, emb in rows], dtype=float)
        yv = np.asarray(y, dtype=int)
        if X.ndim != 2 or X.shape[1] == 0:
            return {**base, "status": "error", "message": "stored embeddings are malformed."}
        face_dims = min(_FACE_DIMS, X.shape[1])
        groups = identity_groups(X[:, :face_dims].tolist(), eps=eps)
        n_groups = len(set(groups))
        base = {**base, "identities": n_groups}

        splits = min(n_splits, n_groups, likes, passes)
        if splits < 2:
            return {**base, "status": "insufficient_groups",
                    "message": f"Too few distinct identities / minority samples for grouped "
                               f"CV (usable folds={splits}). Collect more — ideally distinct people."}

        folds = list(StratifiedGroupKFold(n_splits=splits).split(X, yv, groups=groups))
        roc, pr, brier = [], [], []
        true_accepted = false_dislikes = false_likes = true_dislikes = 0
        for tr, va in folds:
            if len(set(yv[tr].tolist())) < 2 or len(set(yv[va].tolist())) < 2:
                continue                    # a fold without both classes can't be scored
            clf = new_classifier()               # same hyperparameters as the deployed ranker
            clf.fit(X[tr], yv[tr])
            p = clf.predict_proba(X[va])[:, 1]
            roc.append(float(roc_auc_score(yv[va], p)))
            prec, rec, _ = precision_recall_curve(yv[va], p)
            pr.append(float(auc(rec, prec)))
            brier.append(float(brier_score_loss(yv[va], p)))
            predicted_like = p >= threshold
            true_accepted += int(np.sum((yv[va] == 1) & predicted_like))
            false_dislikes += int(np.sum((yv[va] == 1) & ~predicted_like))
            false_likes += int(np.sum((yv[va] == 0) & predicted_like))
            true_dislikes += int(np.sum((yv[va] == 0) & ~predicted_like))
        if not roc:
            return {**base, "status": "no_folds",
                    "message": "No scorable folds (each lacked both classes). Collect more labels."}

        def ms(v):
            return [float(np.mean(v)), float(np.std(v))]
        accepted_total = true_accepted + false_dislikes
        accepted_recall = true_accepted / accepted_total if accepted_total else None
        return {**base, "status": "ok", "folds": len(roc),
                "roc_auc": ms(roc), "pr_auc": ms(pr), "brier": ms(brier),
                "accepted_recall": accepted_recall,
                "false_dislike_rate": (1.0 - accepted_recall) if accepted_recall is not None else None,
                "confusion": {
                    "true_accepted": true_accepted,
                    "false_dislikes": false_dislikes,
                    "false_likes": false_likes,
                    "true_dislikes": true_dislikes,
                },
                "message": f"identity-grouped {len(roc)}-fold CV"}
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": "error", "message": f"evaluation failed: {type(exc).__name__}: {exc}"}


def _capitalize_first_letter(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s


def format_report(r: dict) -> str:
    """Render an evaluate() result for the terminal — headline metrics with calibration context."""
    if r.get("status") != "ok":
        return _capitalize_first_letter(str(r.get("message", "Evaluation unavailable")))
    roc = r.get("roc_auc") or [0.0, 0.0]
    brier = r.get("brier") or [0.0, 0.0]
    acc, band = roc[0] * 100, roc[1] * 100
    base_rate = r.get("base_rate", 0.0)
    kw = SKLEARN_LOGREG_KWARGS
    return (f"Labels={r['labels']}  likes={r['likes']}  passes={r['passes']}  "
            f"distinct identities={r['identities']}\n"
            f"Identity-grouped {r['folds']}-fold CV "
            f"(LogReg C={kw['C']}, class_weight={kw['class_weight']}):\n"
            f"  Accuracy: {acc:.3f}% +/- {band:.3f}%  "
            f"(ROC-AUC concordance — ranks a like above a pass; 50% = random, 100% = perfect)\n"
            f"  Brier: {brier[0]:.3f} +/- {brier[1]:.3f}  "
            f"(calibration + accuracy combined; lower is better, 0.25 ≈ random at 50% base rate)\n"
            f"  Base rate: {base_rate:.1%} likes in training labels")
