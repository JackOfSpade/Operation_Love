"""Leakage-free evaluation of the personal ranker.

Loads your stored (liked, embedding) labels, clusters them by FACE identity — the
first 512 dims of each 1280-d vector are the L2-normalized ArcFace template — so the
same person never straddles a train/test split, then runs identity-grouped,
stratified K-fold cross-validation of the LogisticRegression ranker and reports an
honest ROC-AUC / PR-AUC / Brier.

    python -m tools.eval_aggregation                      # uses config.yaml
    python -m tools.eval_aggregation --config x.yaml --splits 5

Why grouping matters: the same face can appear in multiple profiles (re-scrapes,
profile edits, duplicate accounts). If those land on both sides of a split, ArcFace
lets the model memorize the face and the score is inflated. Grouping by identity
gives a deployment-realistic estimate. (Per the aggregation research, 2026-06.)

Needs the `ml` extra (scikit-learn). It evaluates the CURRENT aggregation as stored;
comparing alternative poolings would require re-embedding the archived photos.
"""
from __future__ import annotations

import argparse

_FACE_DIMS = 512   # first 512 of the 1280-d vector = L2-normed ArcFace identity template


def identity_groups(face_vectors: list[list[float]], eps: float = 0.5) -> list[int]:
    """Cluster rows by face identity via DBSCAN on cosine distance (eps=0.5 ->
    cosine similarity >= 0.5, the buffalo_l same-identity threshold). min_samples=1
    so a face with no near neighbor becomes its own singleton group. Returns one int
    group id per input row."""
    if not face_vectors:
        return []
    import numpy as np
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=eps, min_samples=1, metric="cosine").fit_predict(
        np.asarray(face_vectors, dtype=float))
    return labels.tolist()


def evaluate(samples: list[tuple[bool, list[float]]], n_splits: int = 5, eps: float = 0.5) -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    n = len(samples)
    y = np.asarray([1 if liked else 0 for liked, _ in samples], dtype=int)
    likes, passes = int(y.sum()), n - int(y.sum()) if n else 0
    if n < 10 or len(set(y.tolist())) < 2:
        print(f"Need >=10 labels with BOTH classes to evaluate "
              f"(have {n}: {likes} like / {passes} pass). Seed more in observe mode.")
        return

    X = np.asarray([emb for _, emb in samples], dtype=float)
    face_dims = min(_FACE_DIMS, X.shape[1])
    groups = identity_groups(X[:, :face_dims].tolist(), eps=eps)
    n_groups = len(set(groups))
    print(f"labels={n}  likes={likes}  passes={passes}  "
          f"distinct identities={n_groups}  (merged {n - n_groups} duplicate-identity row(s))")

    splits = min(n_splits, n_groups, min(likes, passes))
    if splits < 2:
        print(f"Too few identity groups / minority-class samples for grouped CV "
              f"(usable splits={splits}). Collect more — ideally distinct people.")
        return

    roc, pr, brier = [], [], []
    try:
        folds = list(StratifiedGroupKFold(n_splits=splits).split(X, y, groups=groups))
    except ValueError as exc:
        print(f"Could not build grouped folds ({exc}). Collect more labels.")
        return
    for tr, va in folds:
        if len(set(y[tr].tolist())) < 2 or len(set(y[va].tolist())) < 2:
            continue                       # a fold without both classes can't be scored
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])[:, 1]
        roc.append(roc_auc_score(y[va], p))
        prec, rec, _ = precision_recall_curve(y[va], p)
        pr.append(auc(rec, prec))
        brier.append(brier_score_loss(y[va], p))

    if not roc:
        print("No scorable folds (each fold lacked both classes). Collect more labels.")
        return

    def stat(v):
        return f"{np.mean(v):.3f} +/- {np.std(v):.3f}"
    print(f"\nidentity-grouped {len(roc)}-fold CV  (LogReg C=0.1, class_weight=balanced):")
    print(f"  ROC-AUC : {stat(roc)}   (0.5 = chance, 1.0 = perfect)")
    print(f"  PR-AUC  : {stat(pr)}    (vs base rate {likes / n:.2f})")
    print(f"  Brier   : {stat(brier)}    (lower is better; calibration)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval_aggregation",
                                 description="Leakage-free, identity-grouped CV of the ranker.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--splits", type=int, default=5, help="max CV folds (clamped to your data)")
    ap.add_argument("--eps", type=float, default=0.5,
                    help="DBSCAN cosine-distance eps for identity grouping (0.5 = sim>=0.5)")
    args = ap.parse_args()

    from operation_love import config as cfg_mod
    from operation_love.ranker import make_store
    cfg = cfg_mod.load(args.config)
    store = make_store(cfg)
    try:
        samples = store.load_labels()
    finally:
        store.close()
    print(f"[eval] loaded {len(samples)} label(s) from {cfg.storage.backend}\n")
    evaluate(samples, n_splits=args.splits, eps=args.eps)


if __name__ == "__main__":
    main()
