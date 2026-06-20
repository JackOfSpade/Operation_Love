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

_FACE_DIMS = 512   # first 512 of the 1280-d vector = L2-normed ArcFace identity template


def identity_groups(face_vectors: list[list[float]], eps: float = 0.5) -> list[int]:
    """Cluster rows by face identity via DBSCAN on cosine distance (eps=0.5 -> cosine
    similarity >= 0.5, the buffalo_l same-identity threshold). min_samples=1 so a face
    with no near neighbor becomes its own singleton group. One int id per input row."""
    if not face_vectors:
        return []
    import numpy as np
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=eps, min_samples=1, metric="cosine").fit_predict(
        np.asarray(face_vectors, dtype=float)).tolist()


def evaluate(samples: list[tuple[bool, list[float]]], n_splits: int = 5, eps: float = 0.5) -> dict:
    """Identity-grouped, stratified K-fold CV. Returns a JSON-able dict: status,
    label counts, distinct identities, and (when ok) ROC-AUC / PR-AUC / Brier as
    [mean, std]. Never raises — failure modes come back as a status + message."""
    n = len(samples)
    y = [1 if liked else 0 for liked, _ in samples]
    likes, passes = sum(y), n - sum(y)
    base = {
        "labels": n, "likes": likes, "passes": passes, "identities": None,
        "folds": 0, "roc_auc": None, "pr_auc": None, "brier": None,
        "base_rate": (likes / n) if n else 0.0,
    }
    if n < 10 or len(set(y)) < 2:
        return {**base, "status": "insufficient_data",
                "message": f"Need ≥10 labels with both like & pass "
                           f"(have {n}: {likes} like / {passes} pass)."}
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_auc_score
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": "no_sklearn", "message": f"scikit-learn unavailable: {exc}"}

    # Wrapped so this never raises (malformed/ragged embeddings, sklearn edge cases):
    # the GUI relies on a status dict, and the CLI calls evaluate() with no guard.
    try:
        X = np.asarray([emb for _, emb in samples], dtype=float)
        yv = np.asarray(y, dtype=int)
        if X.ndim != 2 or X.shape[1] == 0:
            return {**base, "status": "error", "message": "stored embeddings are malformed."}
        face_dims = min(_FACE_DIMS, X.shape[1])
        groups = identity_groups(X[:, :face_dims].tolist(), eps=eps)
        n_groups = len(set(groups))
        base = {**base, "identities": n_groups}

        splits = min(n_splits, n_groups, min(likes, passes))
        if splits < 2:
            return {**base, "status": "insufficient_groups",
                    "message": f"Too few distinct identities / minority samples for grouped "
                               f"CV (usable folds={splits}). Collect more — ideally distinct people."}

        folds = list(StratifiedGroupKFold(n_splits=splits).split(X, yv, groups=groups))
        roc, pr, brier = [], [], []
        for tr, va in folds:
            if len(set(yv[tr].tolist())) < 2 or len(set(yv[va].tolist())) < 2:
                continue                    # a fold without both classes can't be scored
            clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
            clf.fit(X[tr], yv[tr])
            p = clf.predict_proba(X[va])[:, 1]
            roc.append(float(roc_auc_score(yv[va], p)))
            prec, rec, _ = precision_recall_curve(yv[va], p)
            pr.append(float(auc(rec, prec)))
            brier.append(float(brier_score_loss(yv[va], p)))
        if not roc:
            return {**base, "status": "no_folds",
                    "message": "No scorable folds (each lacked both classes). Collect more labels."}

        def ms(v):
            return [float(np.mean(v)), float(np.std(v))]
        return {**base, "status": "ok", "folds": len(roc),
                "roc_auc": ms(roc), "pr_auc": ms(pr), "brier": ms(brier),
                "message": f"identity-grouped {len(roc)}-fold CV"}
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": "error", "message": f"evaluation failed: {type(exc).__name__}: {exc}"}


def format_report(r: dict) -> str:
    """Render an evaluate() result for the terminal."""
    if r.get("status") != "ok":
        return r.get("message", "evaluation unavailable")

    def f(m):
        return f"{m[0]:.3f} +/- {m[1]:.3f}"
    return (f"labels={r['labels']}  likes={r['likes']}  passes={r['passes']}  "
            f"distinct identities={r['identities']}\n"
            f"identity-grouped {r['folds']}-fold CV (LogReg C=0.1, class_weight=balanced):\n"
            f"  ROC-AUC : {f(r['roc_auc'])}   (0.5 = chance, 1.0 = perfect)\n"
            f"  PR-AUC  : {f(r['pr_auc'])}    (base rate {r['base_rate']:.2f})\n"
            f"  Brier   : {f(r['brier'])}    (lower is better; calibration)")
