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
_MARGINAL_RETURN_BATCH = 20
_MARGINAL_RETURN_MIN_LABELS = 25
_MARGINAL_RETURN_FRACTIONS = (0.50, 0.65, 0.80, 1.00)
_MARGINAL_RETURN_REPEATS = 4
_MARGINAL_RETURN_SEED = 1327


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


def _marginal_unavailable(batch: int, message: str, status: str = "too_early") -> dict:
    return {
        "status": status,
        "batch": batch,
        "marginal_return": None,
        "within_noise": None,
        "confidence": "low",
        "message": message,
    }


def _coerce_batch(batch) -> int:
    try:
        batch = int(batch)
    except Exception:  # noqa: BLE001
        return _MARGINAL_RETURN_BATCH
    return batch if batch > 0 else _MARGINAL_RETURN_BATCH


def _float_or_none(value) -> float | None:
    try:
        value = float(value)
    except Exception:  # noqa: BLE001
        return None
    try:
        import math
        return value if math.isfinite(value) else None
    except Exception:  # noqa: BLE001
        return None


def _identity_subset_indices(groups, yv, target_n: int, rng):
    """Pick whole identity groups until target_n rows are included and both classes remain."""
    import numpy as np

    unique = np.asarray(sorted(set(groups.tolist())))
    selected: list[int] = []
    for gid in rng.permutation(unique):
        selected.extend(np.flatnonzero(groups == gid).tolist())
        if len(selected) >= target_n and len(set(yv[selected].tolist())) >= 2:
            break
    if len(selected) < target_n or len(set(yv[selected].tolist())) < 2:
        return None
    return sorted(selected)


def marginal_return_summary(
    samples: list[tuple[bool, list[float]]],
    batch: int = _MARGINAL_RETURN_BATCH,
    n_splits: int = 5,
    eps: float = 0.5,
    min_labels: int = _MARGINAL_RETURN_MIN_LABELS,
    repeats: int = _MARGINAL_RETURN_REPEATS,
    seed: int = _MARGINAL_RETURN_SEED,
    eval_result: dict | None = None,
) -> dict:
    """Estimate PR-AUC gain from the next `batch` labels using a grouped learning curve.

    Returns a JSON-able status dict and never raises. The learning-curve points are
    identity-group subsamples, then each point is scored by `evaluate()` so the CV
    metric math stays shared with the primary model-quality report.
    """
    batch = _coerce_batch(batch)
    try:
        rows = list(samples) if samples is not None else []
    except Exception:  # noqa: BLE001
        return _marginal_unavailable(
            batch, "diminishing-returns estimate failed: samples are malformed.", status="error"
        )

    n = len(rows)
    if n < min_labels:
        return _marginal_unavailable(
            batch, f"Need about {min_labels} labels before estimating marginal return."
        )

    try:
        full = eval_result if eval_result is not None else evaluate(rows, n_splits=n_splits, eps=eps)
        if full.get("status") != "ok":
            # A hard failure (no sklearn / malformed data) won't be fixed by more labels;
            # mark it error so the indicator shows a dash rather than a cold-start value.
            hard = full.get("status") in ("no_sklearn", "error")
            return _marginal_unavailable(
                batch, "Need a scorable identity-grouped CV before estimating marginal return.",
                status="error" if hard else "too_early",
            )
        pr_full = full.get("pr_auc") or []
        full_pr = _float_or_none(pr_full[0] if len(pr_full) > 0 else None)
        pr_std = _float_or_none(pr_full[1] if len(pr_full) > 1 else None)
        if full_pr is None or pr_std is None:
            return _marginal_unavailable(
                batch, "Need PR-AUC mean and fold noise before estimating marginal return.",
                status="error",
            )

        import numpy as np

        yv = np.asarray([1 if liked else 0 for liked, _ in rows], dtype=int)
        X = np.asarray([emb for _, emb in rows], dtype=float)
        if X.ndim != 2 or X.shape[1] == 0 or len(set(yv.tolist())) < 2:
            return _marginal_unavailable(
                batch, "Need well-formed embeddings with both like and pass labels.",
                status="error",
            )
        face_dims = min(_FACE_DIMS, X.shape[1])
        groups = np.asarray(identity_groups(X[:, :face_dims].tolist(), eps=eps))
        if groups.shape[0] != n or len(set(groups.tolist())) < 2:
            return _marginal_unavailable(
                batch, "diminishing-returns estimate failed: identity grouping contradicted full CV.",
                status="error",
            )

        rng = np.random.default_rng(seed)
        repeats = max(1, int(repeats))
        raw_points = []
        for frac in _MARGINAL_RETURN_FRACTIONS:
            target_n = int(np.ceil(n * frac))
            target_n = max(10, min(n, target_n))
            per_size = []
            per_labels = []
            tries = 1 if target_n >= n else repeats
            for _ in range(tries):
                if target_n >= n:
                    ev = full
                    actual_n = n
                else:
                    idx = _identity_subset_indices(groups, yv, target_n, rng)
                    if idx is None:
                        continue
                    subset = [rows[i] for i in idx]
                    actual_n = len(subset)
                    ev = evaluate(subset, n_splits=n_splits, eps=eps)
                if ev.get("status") != "ok":
                    continue
                pr = ev.get("pr_auc") or []
                pr_mean = _float_or_none(pr[0] if len(pr) > 0 else None)
                if pr_mean is None:
                    continue
                per_size.append(pr_mean)
                per_labels.append(float(actual_n))
            if per_size:
                raw_points.append({
                    "labels": float(np.mean(per_labels)),
                    "pr_auc": float(np.mean(per_size)),
                    "repeats": len(per_size),
                })

        by_size: dict[int, list[dict]] = {}
        for point in raw_points:
            key = int(round(point["labels"]))
            by_size.setdefault(key, []).append(point)
        curve = []
        for size, points in by_size.items():
            curve.append({
                "labels": size,
                "pr_auc": float(np.mean([p["pr_auc"] for p in points])),
                "repeats": int(sum(p["repeats"] for p in points)),
            })
        curve.sort(key=lambda p: p["labels"])

        usable = [p for p in curve if p["labels"] > 0 and _float_or_none(p["pr_auc"]) is not None]
        if len(usable) < 2:
            return _marginal_unavailable(
                batch, "Need more usable learning-curve points before estimating marginal return."
            )

        ns = np.asarray([float(p["labels"]) for p in usable], dtype=float)
        prs = np.asarray([float(p["pr_auc"]) for p in usable], dtype=float)

        # Whole-curve reciprocal fit pr ≈ a - b/n (robust to a single noisy point); the
        # projected gain from the next `batch` labels is then b*batch/(n*(n+batch)).
        recip = None
        fit_method = "reciprocal_fit"
        fit_r2 = 0.0
        if len(usable) >= 3:
            try:
                xs = 1.0 / ns
                slope, intercept = np.polyfit(xs, prs, 1)
                b = -float(slope)
                pred = slope * xs + intercept
                ss_res = float(np.sum((prs - pred) ** 2))
                ss_tot = float(np.sum((prs - float(np.mean(prs))) ** 2))
                fit_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
                recip = _float_or_none(b * batch / (n * (n + batch)))
            except Exception:  # noqa: BLE001
                recip = None

        # Local secant over the top two points as fallback / cross-check.
        prev, cur = usable[-2], usable[-1]
        dn = float(cur["labels"] - prev["labels"])
        fd = _float_or_none(((cur["pr_auc"] - prev["pr_auc"]) / dn) * batch) if dn > 0 else None
        if recip is None:
            fit_method = "finite_difference"

        # ONE always-live value: prefer the positive whole-curve estimate; when noise
        # dominates (estimate <=0), report the magnitude instead so the number stays
        # a tiny positive that keeps shrinking: it decays toward, but never reaches, zero
        # (no cap / no "done"). The user reads the magnitude itself as the degree of
        # diminishing returns; smaller = more diminished.
        gain = next((c for c in (recip, fd) if c is not None and c > 0), None)
        if gain is None:
            gain = abs(recip) if recip else (abs(fd) if fd else None)
        gain = _float_or_none(gain)
        if gain is None or gain <= 0:
            gain = 1e-6   # floor: the indicator is never exactly zero (never "complete")

        within_noise = bool(gain < pr_std)
        if (fit_method != "reciprocal_fit" or within_noise or len(usable) < 4 or
                fit_r2 < 0.4 or n < 50 or pr_std > 0.10):
            confidence = "low"
        elif n >= 100 and pr_std <= 0.04 and fit_r2 >= 0.8:
            confidence = "high"
        else:
            confidence = "med"

        return {
            "status": "ok",
            "batch": batch,
            "labels": n,
            "marginal_return": float(gain),
            "within_noise": within_noise,
            "confidence": confidence,
            "message": f"projected PR-AUC gain from the next {batch} labels",
            "curve": usable,
            "fit": {"method": fit_method, "r2": float(fit_r2)},
        }
    except Exception as exc:  # noqa: BLE001
        return _marginal_unavailable(
            batch, f"marginal return failed: {type(exc).__name__}: {exc}", status="error"
        )


def _format_marginal_value(value) -> str | None:
    """Format a small positive gain in decimal notation with enough places to keep ~2
    significant figures, so it never rounds to zero as it shrinks (0.014 -> 0.0021 ->
    0.00038 -> ...). Falls back to scientific only when absurdly small."""
    value = _float_or_none(value)
    if value is None:
        return None
    import math
    v = abs(value)
    if v == 0:
        return "0"
    places = max(2, 1 - math.floor(math.log10(v)))
    if places > 9:
        return f"{v:.2e}"
    return f"{v:.{places}f}".rstrip("0").rstrip(".")


def _format_marginal_return(m: dict | None) -> str | None:
    """One indicator, value only — no qualifier wording. The magnitude is the signal:
    smaller number = deeper into diminishing returns."""
    if not m:
        return None
    if m.get("status") != "ok":
        return "diminishing returns —"
    value = _format_marginal_value(m.get("marginal_return"))
    if value is None:
        return "diminishing returns —"
    return f"diminishing returns {value} PR-AUC / +{m.get('batch', _MARGINAL_RETURN_BATCH)} labels"


def format_report(r: dict) -> str:
    """Render an evaluate() result for the terminal."""
    if r.get("status") != "ok":
        msg = r.get("message", "evaluation unavailable")
        marginal = _format_marginal_return(r.get("marginal_return"))
        return msg if not marginal else f"{msg}\n  {marginal}"

    def f(m):
        return f"{m[0]:.3f} +/- {m[1]:.3f}"
    marginal = _format_marginal_return(r.get("marginal_return"))
    marginal_line = f"\n  {marginal}" if marginal else ""
    return (f"labels={r['labels']}  likes={r['likes']}  passes={r['passes']}  "
            f"distinct identities={r['identities']}\n"
            f"identity-grouped {r['folds']}-fold CV (LogReg C=0.1, class_weight=balanced):\n"
            f"  Area under the precision-recall curve: {f(r['pr_auc'])}  "
            f"(base rate {r['base_rate']:.2f}; primary metric here — watch lift over base)\n"
            f"  Area under the receiver-operating-characteristic curve: {f(r['roc_auc'])}  "
            f"(0.5 = chance, 1.0 = perfect)\n"
            f"  Brier score: {f(r['brier'])}  (lower is better; calibration)"
            f"{marginal_line}")
