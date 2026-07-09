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

from .model import SKLEARN_LOGREG_KWARGS, new_classifier   # shared ranker hyperparameters (no drift)

_FACE_DIMS = 512   # first 512 of the 1280-d vector = L2-normed ArcFace identity template
_IDENTITY_EPS = 0.5   # DBSCAN cosine-distance threshold -> cosine similarity >= 0.5,
                      # the buffalo_l same-identity threshold (see identity_groups below)


def identity_groups(face_vectors: list[list[float]], eps: float = _IDENTITY_EPS) -> list[int]:
    """Cluster rows by face identity via DBSCAN on cosine distance (eps=0.5 -> cosine
    similarity >= 0.5, the buffalo_l same-identity threshold). min_samples=1 so a face
    with no near neighbor becomes its own singleton group. One int id per input row."""
    if not face_vectors:
        return []
    import numpy as np
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=eps, min_samples=1, metric="cosine").fit_predict(
        np.asarray(face_vectors, dtype=float)).tolist()


def evaluate(samples: list[tuple[bool, list[float]]], n_splits: int = 5,
             eps: float = _IDENTITY_EPS) -> dict:
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
            clf = new_classifier()               # same hyperparameters as the deployed ranker
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


def quality_trajectory(samples, step: int = 5, n_splits: int = 5, eps: float = _IDENTITY_EPS,
                       min_labels: int = 10, max_points: int = 40) -> list[dict]:
    """Recompute leakage-free grouped-CV accuracy at chronological prefixes (every
    `step` labels) so the hub can chart how ranking quality evolved as labels accumulated.

    `samples` MUST be in swipe (created_at) order — prefix [:k] is then "the first k
    labels you collected". The effective step is widened so at most ~`max_points` prefixes
    are scored, bounding cost (each prefix is a full grouped CV) as labels grow. Returns a
    JSON-able list of points, only where grouped CV is valid (early prefixes with too few
    identities/classes are skipped). Never raises.
    """
    try:
        rows = list(samples) if samples else []
    except Exception:  # noqa: BLE001
        return []
    n = len(rows)
    step = max(1, int(step))
    if max_points and max_points > 0:
        step = max(step, -(-n // max_points))        # ceil(n/max_points): coarsen so points <= ~max_points
    sizes = list(range(step, n + 1, step))
    if n >= min_labels and (not sizes or sizes[-1] != n):
        sizes.append(n)                              # always include the full set as the last point
    points: list[dict] = []
    for size in sizes:
        if size < min_labels:
            continue
        try:
            r = evaluate(rows[:size], n_splits=n_splits, eps=eps)
        except Exception:  # noqa: BLE001
            continue
        if r.get("status") != "ok":
            continue
        pr = r.get("pr_auc") or [None, None]
        roc = r.get("roc_auc") or [None, None]
        brier = r.get("brier") or [None, None]
        points.append({
            "labels": size, "identities": r.get("identities"),
            "pr_auc": pr[0], "pr_std": pr[1],
            "roc_auc": roc[0], "roc_std": roc[1], "brier": brier[0],
            "base_rate": r.get("base_rate"),
        })
    return points


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
