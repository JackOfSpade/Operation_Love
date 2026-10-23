"""Decide, by a pre-registered rule, whether the ranker switches to the whole-photo feature set.

    python -m tools.compare_feature_sets --config config.yaml
    python -m tools.compare_feature_sets --config config.yaml --seeds 20 --orders 20 --jobs 4

PRE-REGISTERED SWITCH RULE (fixed before the live comparison was run; a result never edits it):
switch to `face_crop_whole_photo_v1` only if ALL hold: (1) the corrected-t 95% CI of the paired
ΔAUC has its lower bound > 0; (2) Δ>0 on at least 95% of seeds; (3) the mean nested
dislikes_passed at the configured recall target is not lower than v1's; (4) the mean nested
accepted_recall is not lower than v1's by more than 0.01. Otherwise keep `face_crop_v1`.

WHAT IS COMPARED. The same labels, in the same order, under both feature sets:
`fetch_label_rows(store, "face_crop_v1")` (today's 1280-d vector) and `fetch_label_rows(store,
"face_crop_whole_photo_v1")` (that vector ++ the 768-d pooled whole-photo vector). The tool
refuses when the two disagree about any label (count, like/pass, source, or a v1 prefix that is
not byte-identical), and when any label still lacks its whole-photo row (run
`python -m tools.backfill_photo_features` first). Rows are put in a content-hash order first, so
every number below is a function of the label SET, not of the order the store returns it in.
All labels take part, review answers included, exactly as the model-quality card evaluates them.

WHAT IS MEASURED (the shipped classifier `new_classifier()` throughout; identity groups are
`identity_groups` on dims 0-511, the ArcFace template both feature sets share, so one person
never straddles a split and both arms see the SAME splits):
1. AUC. StratifiedGroupKFold(5, shuffle=True, random_state=seed) for seed 0..seeds-1. Per seed:
   the pooled out-of-fold AUC of each arm and their paired difference Δ (clause 2 counts the
   seeds with Δ>0). Per split j (k=5 folds x r=seeds repetitions): d_j = AUC_wp - AUC_v1 on
   that fold's held-out people. The Nadeau-Bengio corrected repeated-CV t interval is
       mean(d) +/- t_{0.975, kr-1} * sqrt((1/(k*r) + n_test/n_train) * s^2)
   with s^2 the sample variance (ddof=1) of the kr deltas and n_test/n_train the mean of the
   per-split held-out/training size ratios (group folds are not exactly equal). The naive
   1/(kr) alone ignores that every repetition re-uses the same people, which makes the deltas
   strongly correlated and the naive interval far too narrow; the n_test/n_train term is the
   correction (Nadeau & Bengio 2003; Bouckaert & Frank 2004).
2. Nested recall-target evaluation. `evaluate(samples, n_splits=5,
   recall_target=ranker.like_recall_target)`, the model-quality card's own function (the whole
   calibration re-run inside each fold, held-out people scored at the resulting cutoff), averaged
   over `--orders` row permutations (its outer folds are unshuffled, so order is what varies
   them; permutation o uses numpy default_rng(o), the same for both arms). Reported:
   dislikes_passed, accepted_recall, share of the deck liked, unwanted likes and lost likes per
   100 profiles.
3. Time-forward (diagnostic, not part of the rule). Labels sorted by created_at; train on the
   oldest 50/60/70/80%, test on the rest minus anyone whose identity group appears in training;
   scorer = the deployed fold ensemble (`fit_folds`) fitted on the training side. Label
   timestamps are read with a read-only query through the backend's own retraction predicate
   and matched to the loaded rows by (liked, source, v1 vector); when that is impossible the
   section says why and the verdict is unaffected.

OUTPUT is aggregates only: never a profile id, a name or a vector.

EXIT STATUS: 0 = SWITCH, 3 = KEEP, 2 = the comparison could not be made (it says why).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

V1 = "face_crop_v1"
WHOLE_PHOTO = "face_crop_whole_photo_v1"
ARMS = (V1, WHOLE_PHOTO)

SWITCH_RULE = (
    "switch to `face_crop_whole_photo_v1` only if ALL hold: (1) the corrected-t 95% CI of the "
    "paired ΔAUC has its lower bound > 0; (2) Δ>0 on at least 95% of seeds; (3) the mean nested "
    "dislikes_passed at the configured recall target is not lower than v1's; (4) the mean "
    "nested accepted_recall is not lower than v1's by more than 0.01. Otherwise keep "
    "`face_crop_v1`."
)

# Pre-registered constants.  The rule above is stated in terms of these; changing one is
# changing the rule.
DEFAULT_SEEDS = 20
DEFAULT_ORDERS = 20
N_SPLITS = 5
CONFIDENCE = 0.95
SEED_SHARE_PERCENT = 95
RECALL_SLACK = 0.01
TIME_CUTS = (0.5, 0.6, 0.7, 0.8)
# Absorbs binary-float noise at the rule's two equality boundaries (a mean recall that is
# exactly 0.01 lower must pass clause 4, not fail on 0.010000000000000009).
_FLOAT_TOL = 1e-12

EXIT_SWITCH = 0
EXIT_REFUSED = 2
EXIT_KEEP = 3


class ComparisonRefused(Exception):
    """The comparison cannot be made honestly with these labels (reason in args[0])."""


# --------------------------------------------------------------------------- statistics


def corrected_repeated_cv_t(
    deltas: Sequence[float],
    *,
    n_test: Sequence[int],
    n_train: Sequence[int],
    confidence: float = CONFIDENCE,
) -> dict:
    """Nadeau-Bengio corrected repeated k-fold t interval of the mean paired difference.

    ``deltas`` holds one paired difference per split (k folds x r repetitions, so k*r values);
    ``n_test`` / ``n_train`` the held-out and training sizes of those same splits.  With J = k*r,
    ``s2`` the sample variance (ddof=1) of the deltas and ``rho`` the mean of n_test/n_train::

        se = sqrt((1/J + rho) * s2)
        CI = mean +/- t_{(1+confidence)/2, J-1} * se

    A zero variance gives a zero-width interval at the mean (and ``t`` None).
    """
    import numpy as np
    from scipy.stats import t as student_t

    d = np.asarray(deltas, dtype=float)
    te = np.asarray(n_test, dtype=float)
    tr = np.asarray(n_train, dtype=float)
    if d.ndim != 1 or d.size < 2:
        raise ValueError("the corrected t interval needs at least 2 split deltas")
    if te.shape != d.shape or tr.shape != d.shape:
        raise ValueError("n_test and n_train need one size per split delta")
    if not np.isfinite(d).all():
        raise ValueError("split deltas must be finite")
    if (te <= 0).any() or (tr <= 0).any():
        raise ValueError("split sizes must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    splits = int(d.size)
    mean = float(d.mean())
    variance = float(d.var(ddof=1))
    ratio = float(np.mean(te / tr))
    se = math.sqrt((1.0 / splits + ratio) * variance)
    df = splits - 1
    t_crit = float(student_t.ppf(0.5 + confidence / 2.0, df))
    return {
        "splits": splits,
        "mean": mean,
        "variance": variance,
        "test_train_ratio": ratio,
        "se": se,
        "df": df,
        "t_crit": t_crit,
        "t": mean / se if se > 0.0 else None,
        "lower": mean - t_crit * se,
        "upper": mean + t_crit * se,
        "confidence": confidence,
    }


@dataclass(frozen=True)
class Verdict:
    switch: bool
    clauses: dict  # clause number -> bool, in rule order


def apply_switch_rule(
    *,
    ci_lower: float,
    seeds_positive: int,
    seeds: int,
    dislikes_passed_v1: float,
    dislikes_passed_wp: float,
    recall_v1: float,
    recall_wp: float,
) -> Verdict:
    """The PRE-REGISTERED SWITCH RULE (see the module docstring), clause by clause."""
    if type(seeds) is not int or seeds < 1:
        raise ValueError("seeds must be a positive integer")
    if type(seeds_positive) is not int or not 0 <= seeds_positive <= seeds:
        raise ValueError("seeds_positive must be an integer in [0, seeds]")
    clauses = {
        1: ci_lower > 0.0,
        2: seeds_positive * 100 >= SEED_SHARE_PERCENT * seeds,
        3: dislikes_passed_wp >= dislikes_passed_v1 - _FLOAT_TOL,
        4: recall_v1 - recall_wp <= RECALL_SLACK + _FLOAT_TOL,
    }
    return Verdict(switch=all(clauses.values()), clauses=clauses)


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Paired:
    """Both feature sets' rows for the same labels, in canonical (content-hash) order."""

    y: object  # numpy int array, 1 = liked
    x_v1: object  # numpy (n, d1)
    x_wp: object  # numpy (n, d2), d2 > d1, x_wp[:, :d1] == x_v1
    manual: int
    review: int
    original_index: list  # canonical position -> position in the rows as loaded


def _triple(row: object, ordinal: int, name: str) -> tuple[bool, list, str]:
    if not isinstance(row, (list, tuple)) or len(row) != 3:
        raise ComparisonRefused(
            f"{name} row {ordinal} is not a (liked, embedding, source) triple"
        )
    liked, embedding, source = row
    if type(liked) is not bool:
        raise ComparisonRefused(f"{name} row {ordinal} liked label is not a bool")
    if not isinstance(embedding, (list, tuple)) or not embedding:
        raise ComparisonRefused(f"{name} row {ordinal} has no feature vector")
    return liked, list(embedding), "" if source is None else str(source)


def pair_rows(v1_rows: Sequence, wp_rows: Sequence) -> Paired:
    """Validate that both loads describe the SAME labels in the SAME order; canonicalise them."""
    import numpy as np

    from operation_love.ranker.threshold import is_calibration_source

    if len(v1_rows) != len(wp_rows):
        raise ComparisonRefused(
            f"the feature sets returned different label counts ({len(v1_rows)} "
            f"{V1} vs {len(wp_rows)} {WHOLE_PHOTO})"
        )
    if not v1_rows:
        raise ComparisonRefused("there are no labels to compare")
    liked_list: list[bool] = []
    sources: list[str] = []
    v1_vectors: list[list] = []
    wp_vectors: list[list] = []
    for ordinal, (a, b) in enumerate(zip(v1_rows, wp_rows, strict=True)):
        liked, emb1, source = _triple(a, ordinal, V1)
        liked2, emb2, source2 = _triple(b, ordinal, WHOLE_PHOTO)
        if liked != liked2 or source != source2:
            raise ComparisonRefused(
                f"label {ordinal} differs between the feature sets (like/pass or source); "
                "they must be the same labels in the same order"
            )
        v1_vectors.append(emb1)
        wp_vectors.append(emb2)
        liked_list.append(liked)
        sources.append(source)
    try:
        x1 = np.asarray(v1_vectors, dtype=float)
        x2 = np.asarray(wp_vectors, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ComparisonRefused(
            f"feature vectors are ragged or malformed: {exc}"
        ) from exc
    if x1.ndim != 2 or x2.ndim != 2:
        raise ComparisonRefused("feature vectors are ragged")
    if x2.shape[1] <= x1.shape[1]:
        raise ComparisonRefused(
            f"{WHOLE_PHOTO} vectors ({x2.shape[1]}-d) are not longer than {V1} "
            f"({x1.shape[1]}-d)"
        )
    if not (np.isfinite(x1).all() and np.isfinite(x2).all()):
        raise ComparisonRefused("a feature vector has a non-finite value")
    mismatched = np.flatnonzero((x2[:, : x1.shape[1]] != x1).any(axis=1))
    if mismatched.size:
        raise ComparisonRefused(
            f"{mismatched.size} label(s) (first: label {int(mismatched[0])}) have a "
            f"{WHOLE_PHOTO} vector whose first {x1.shape[1]} dims are not the {V1} vector; "
            "they must be the same labels in the same order"
        )
    y = np.asarray(liked_list, dtype=int)
    keys = []
    for i in range(y.shape[0]):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(b"\x01" if y[i] else b"\x00")
        digest.update(np.ascontiguousarray(x2[i], dtype=np.float64).tobytes())
        keys.append(digest.digest())
    order = sorted(range(y.shape[0]), key=keys.__getitem__)
    manual = sum(1 for s in sources if is_calibration_source(s))
    return Paired(
        y=y[order],
        x_v1=x1[order],
        x_wp=x2[order],
        manual=manual,
        review=len(sources) - manual,
        original_index=order,
    )


def _identity_groups(p: Paired):
    import numpy as np

    from operation_love.ranker.evaluate import _FACE_DIMS, identity_groups

    face = p.x_v1[:, : min(_FACE_DIMS, p.x_v1.shape[1])]
    try:
        return np.asarray(identity_groups(face.tolist()))
    except ValueError as exc:
        raise ComparisonRefused(f"identity grouping failed: {exc}") from exc


def _arms(p: Paired):
    return ((V1, p.x_v1), (WHOLE_PHOTO, p.x_wp))


# --------------------------------------------------------------------------- 1. AUC


def cv_comparison(
    p: Paired,
    groups,
    *,
    seeds: int,
    n_splits: int = N_SPLITS,
    classifier_factory: Callable[[], object] | None = None,
) -> dict:
    """Shuffled identity-grouped k-fold OOF AUC for both arms on the SAME splits, per seed."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    if classifier_factory is None:
        from operation_love.ranker.model import new_classifier as classifier_factory

    n = int(p.y.shape[0])
    n_groups = len(set(groups.tolist()))
    likes = int(p.y.sum())
    if min(n_groups, likes, n - likes) < n_splits:
        raise ComparisonRefused(
            f"{n_splits}-fold identity-grouped CV needs at least {n_splits} identities, likes "
            f"and passes (have {n_groups} identities, {likes} liked / {n - likes} passed)"
        )
    seed_auc: dict[str, list[float]] = {arm: [] for arm in ARMS}
    split_deltas: list[float] = []
    n_test: list[int] = []
    n_train: list[int] = []
    for seed in range(seeds):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        folds = list(splitter.split(p.x_v1, p.y, groups=groups))
        oof = {arm: np.full(n, np.nan) for arm in ARMS}
        for tr, va in folds:
            if len(set(p.y[tr].tolist())) < 2 or len(set(p.y[va].tolist())) < 2:
                raise ComparisonRefused(
                    f"seed {seed} produced a fold without both likes and passes; "
                    "collect more labels"
                )
            fold_auc = {}
            for arm, X in _arms(p):
                clf = classifier_factory()
                clf.fit(X[tr], p.y[tr])
                scores = np.asarray(clf.predict_proba(X[va]), dtype=float)[:, 1]
                oof[arm][va] = scores
                fold_auc[arm] = float(roc_auc_score(p.y[va], scores))
            split_deltas.append(fold_auc[WHOLE_PHOTO] - fold_auc[V1])
            n_test.append(len(va))
            n_train.append(len(tr))
        for arm in ARMS:
            seed_auc[arm].append(float(roc_auc_score(p.y, oof[arm])))
    seed_deltas = [
        a - b for a, b in zip(seed_auc[WHOLE_PHOTO], seed_auc[V1], strict=True)
    ]
    return {
        "seeds": seeds,
        "n_splits": n_splits,
        "auc_mean": {arm: float(np.mean(v)) for arm, v in seed_auc.items()},
        "auc_seed_sd": {arm: float(np.std(v)) for arm, v in seed_auc.items()},
        "seed_delta_mean": float(np.mean(seed_deltas)),
        "seeds_positive": int(sum(1 for d in seed_deltas if d > 0.0)),
        "corrected_t": corrected_repeated_cv_t(
            split_deltas, n_test=n_test, n_train=n_train
        ),
    }


# --------------------------------------------------------------------------- 2. nested


def _nested_metrics(result: dict, arm: str, order_index: int) -> dict:
    if result.get("status") != "ok":
        raise ComparisonRefused(
            f"nested evaluation of {arm} (row order {order_index}) is "
            f"{result.get('status')!r}: {result.get('message')}"
        )
    c = result["confusion"]
    total = sum(int(c[k]) for k in c)
    if (
        not total
        or result.get("dislikes_passed") is None
        or result.get("accepted_recall") is None
    ):
        raise ComparisonRefused(
            f"nested evaluation of {arm} (row order {order_index}) scored no likes or no passes"
        )
    return {
        "dislikes_passed": float(result["dislikes_passed"]),
        "accepted_recall": float(result["accepted_recall"]),
        "like_share": (c["true_accepted"] + c["false_likes"]) / total,
        "unwanted_likes_per_100": 100.0 * c["false_likes"] / total,
        "lost_likes_per_100": 100.0 * c["false_dislikes"] / total,
    }


NESTED_METRICS = (
    "dislikes_passed",
    "accepted_recall",
    "like_share",
    "unwanted_likes_per_100",
    "lost_likes_per_100",
)


def nested_comparison(
    p: Paired,
    *,
    orders: int,
    recall_target: float,
    jobs: int = 1,
    evaluate_fn: Callable[..., dict] | None = None,
) -> dict:
    """``evaluate(..., recall_target)`` for both arms over ``orders`` shared row permutations."""
    import numpy as np

    if evaluate_fn is None:
        from operation_love.ranker.evaluate import evaluate as evaluate_fn

    n = int(p.y.shape[0])

    def one(order_index: int) -> dict:
        perm = np.random.default_rng(order_index).permutation(n)
        out = {}
        for arm, X in _arms(p):
            samples = [(bool(p.y[i]), X[i].tolist()) for i in perm]
            result = evaluate_fn(
                samples, n_splits=N_SPLITS, recall_target=recall_target
            )
            out[arm] = _nested_metrics(result, arm, order_index)
        return out

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            per_order = list(pool.map(one, range(orders)))
    else:
        per_order = [one(o) for o in range(orders)]
    return {
        "orders": orders,
        "recall_target": recall_target,
        "mean": {
            arm: {
                metric: float(np.mean([r[arm][metric] for r in per_order]))
                for metric in NESTED_METRICS
            }
            for arm in ARMS
        },
    }


# --------------------------------------------------------------------------- 3. time-forward


def time_forward(
    p: Paired,
    groups,
    times: Sequence[float],
    *,
    cuts: Sequence[float] = TIME_CUTS,
    classifier_factory: Callable[[], object] | None = None,
) -> list[dict]:
    """Train on the oldest share, test on later identity-disjoint people; ΔAUC per cut."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from operation_love.ranker.threshold import fit_folds

    t = np.asarray(times, dtype=float)
    n = int(p.y.shape[0])
    if t.shape != (n,) or not np.isfinite(t).all():
        raise ValueError("time_forward needs one finite timestamp per label")
    order = np.argsort(t, kind="stable")
    out = []
    for cut in cuts:
        n_train = int(round(cut * n))
        tr = order[:n_train]
        seen = set(groups[tr].tolist())
        te = np.asarray(
            [i for i in order[n_train:] if groups[i] not in seen], dtype=int
        )
        row = {"cut": cut, "n_train": int(tr.size), "n_test": int(te.size)}
        row["test_likes"] = int(p.y[te].sum()) if te.size else 0
        if te.size == 0 or len(set(p.y[te].tolist())) < 2:
            row["status"] = "test side lacks likes or passes"
            out.append(row)
            continue
        auc = {}
        for arm, X in _arms(p):
            fit = fit_folds(
                X[tr],
                p.y[tr],
                groups[tr].tolist(),
                classifier_factory=classifier_factory,
            )
            if fit is None:
                break
            scores = fit.ensemble.predict_proba(X[te])[:, 1]
            auc[arm] = float(roc_auc_score(p.y[te], scores))
        if len(auc) != len(ARMS):
            row["status"] = "training side cannot be fold-fitted"
            out.append(row)
            continue
        row.update(
            status="ok",
            auc=auc,
            delta=auc[WHOLE_PHOTO] - auc[V1],
        )
        out.append(row)
    return out


# --------------------------------------------------------------------------- orchestration


def compare(
    v1_rows: Sequence,
    wp_rows: Sequence,
    *,
    seeds: int = DEFAULT_SEEDS,
    orders: int = DEFAULT_ORDERS,
    recall_target: float,
    times: Sequence[float] | None = None,
    time_note: str = "",
    jobs: int = 1,
    classifier_factory: Callable[[], object] | None = None,
) -> dict:
    """Every measurement plus the verdict, as a dict of aggregates (no ids, no vectors)."""
    if type(seeds) is not int or seeds < 1:
        raise ValueError("seeds must be a positive integer")
    if type(orders) is not int or orders < 1:
        raise ValueError("orders must be a positive integer")
    if type(jobs) is not int or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    p = pair_rows(v1_rows, wp_rows)
    groups = _identity_groups(p)
    cv = cv_comparison(p, groups, seeds=seeds, classifier_factory=classifier_factory)
    nested = nested_comparison(p, orders=orders, recall_target=recall_target, jobs=jobs)
    tf: list[dict] | None = None
    if times is not None:
        canonical_times = [times[i] for i in p.original_index]
        tf = time_forward(
            p, groups, canonical_times, classifier_factory=classifier_factory
        )
    verdict = apply_switch_rule(
        ci_lower=cv["corrected_t"]["lower"],
        seeds_positive=cv["seeds_positive"],
        seeds=seeds,
        dislikes_passed_v1=nested["mean"][V1]["dislikes_passed"],
        dislikes_passed_wp=nested["mean"][WHOLE_PHOTO]["dislikes_passed"],
        recall_v1=nested["mean"][V1]["accepted_recall"],
        recall_wp=nested["mean"][WHOLE_PHOTO]["accepted_recall"],
    )
    n = int(p.y.shape[0])
    likes = int(p.y.sum())
    return {
        "labels": n,
        "likes": likes,
        "passes": n - likes,
        "manual": p.manual,
        "review": p.review,
        "identities": len(set(groups.tolist())),
        "dims": {V1: int(p.x_v1.shape[1]), WHOLE_PHOTO: int(p.x_wp.shape[1])},
        "preregistered_run": seeds == DEFAULT_SEEDS and orders == DEFAULT_ORDERS,
        "cv": cv,
        "nested": nested,
        "time_forward": tf,
        "time_note": time_note,
        "verdict": {"switch": verdict.switch, "clauses": verdict.clauses},
    }


def format_report(r: dict) -> str:
    cv, nested = r["cv"], r["nested"]
    ct = cv["corrected_t"]
    width = max(len(arm) for arm in ARMS)
    lines = [
        f"Feature-set comparison: {V1} ({r['dims'][V1]}-d) vs {WHOLE_PHOTO} "
        f"({r['dims'][WHOLE_PHOTO]}-d)",
        f"Labels: {r['labels']} ({r['likes']} like / {r['passes']} pass; {r['manual']} manual, "
        f"{r['review']} auto-review answers), {r['identities']} identities",
        f"Run: {cv['seeds']} seeds x {cv['n_splits']} folds, {nested['orders']} row orders, "
        f"recall target {nested['recall_target']:.2f}"
        + (
            " (the pre-registered configuration)"
            if r["preregistered_run"]
            else f" (NOT the pre-registered --seeds {DEFAULT_SEEDS} --orders "
            f"{DEFAULT_ORDERS}; treat the verdict as exploratory)"
        ),
        "",
        f"1. Identity-grouped {cv['n_splits']}-fold out-of-fold AUC, {cv['seeds']} shuffled seeds",
    ]
    lines.extend(
        f"   {arm:<{width}}  {cv['auc_mean'][arm]:.4f} (seed SD {cv['auc_seed_sd'][arm]:.4f})"
        for arm in ARMS
    )
    lines += [
        f"   paired ΔAUC (pooled OOF, mean over seeds) {cv['seed_delta_mean']:+.4f}; "
        f"Δ>0 on {cv['seeds_positive']}/{cv['seeds']} seeds",
        f"   per-split ΔAUC {ct['mean']:+.4f}, corrected-t {ct['confidence']:.0%} CI "
        f"[{ct['lower']:+.4f}, {ct['upper']:+.4f}] (df {ct['df']}, n_test/n_train "
        f"{ct['test_train_ratio']:.3f}"
        + (f", t {ct['t']:.2f})" if ct["t"] is not None else ", zero variance)"),
        "",
        f"2. Nested recall-target evaluation at {nested['recall_target']:.2f}, mean over "
        f"{nested['orders']} row orders",
        f"   {'':<{width}}  dislikes_passed  accepted_recall  deck liked  "
        "unwanted/100  lost/100",
    ]
    for arm in ARMS:
        m = nested["mean"][arm]
        lines.append(
            f"   {arm:<{width}}  {m['dislikes_passed']:15.4f}  {m['accepted_recall']:15.4f}  "
            f"{m['like_share']:10.1%}  {m['unwanted_likes_per_100']:12.1f}  "
            f"{m['lost_likes_per_100']:8.1f}"
        )
    lines += ["", "3. Time-forward ΔAUC (diagnostic; identity-disjoint later labels)"]
    if r["time_forward"] is None:
        lines.append(f"   unavailable: {r['time_note'] or 'no label timestamps'}")
    else:
        for row in r["time_forward"]:
            head = (
                f"   train oldest {row['cut']:.0%} (n={row['n_train']}), "
                f"test n={row['n_test']} ({row['test_likes']} likes)"
            )
            if row["status"] != "ok":
                lines.append(f"{head}: {row['status']}")
            else:
                lines.append(
                    f"{head}: AUC {row['auc'][V1]:.4f} -> {row['auc'][WHOLE_PHOTO]:.4f} "
                    f"(Δ {row['delta']:+.4f})"
                )
    c = r["verdict"]["clauses"]
    v1m, wpm = nested["mean"][V1], nested["mean"][WHOLE_PHOTO]

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    lines += [
        "",
        "4. Pre-registered switch rule",
        f"   (1) corrected-t CI lower bound {ct['lower']:+.4f} > 0: {mark(c[1])}",
        f"   (2) Δ>0 on {cv['seeds_positive']}/{cv['seeds']} seeds (needs >= "
        f"{SEED_SHARE_PERCENT}%): {mark(c[2])}",
        f"   (3) dislikes_passed {wpm['dislikes_passed']:.4f} vs {V1} "
        f"{v1m['dislikes_passed']:.4f} (not lower): {mark(c[3])}",
        f"   (4) accepted_recall {wpm['accepted_recall']:.4f} vs {V1} "
        f"{v1m['accepted_recall']:.4f} (at most {RECALL_SLACK} lower): {mark(c[4])}",
        "",
        (
            f"VERDICT: SWITCH to {WHOLE_PHOTO} (set ranker.feature_set: {WHOLE_PHOTO})"
            if r["verdict"]["switch"]
            else f"VERDICT: KEEP {V1}"
        ),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- store access


def load_rows(store) -> tuple[list, list]:
    """Both feature sets' label rows; a label without a whole-photo row refuses the comparison."""
    from operation_love.ranker.threshold import FeatureSetIncomplete, fetch_label_rows

    v1_rows = list(fetch_label_rows(store, V1))
    try:
        wp_rows = list(fetch_label_rows(store, WHOLE_PHOTO))
    except FeatureSetIncomplete as exc:
        # The loader's own sentence: the counts (stored zero-card rows included) and the
        # backfill command, the same one every other surface prints.
        raise ComparisonRefused(exc.reason) from exc
    return v1_rows, wp_rows


def _row_key(liked: object, embedding: Sequence, source: object) -> tuple:
    return (
        bool(liked),
        "" if source is None else str(source),
        tuple(float(v) for v in embedding),
    )


def _stamped_label_rows(store) -> list[tuple] | None:
    """(liked, v1 embedding, source, created_at epoch seconds) for every non-retracted label.

    Read-only, through each backend's own retraction predicate, so it returns the same label
    set ``load_label_rows`` does.  ``None`` for a store this tool cannot query.
    """
    from operation_love.ranker.bigquery_store import BigQueryStore
    from operation_love.ranker.store import _LABEL_NOT_RETRACTED, SQLiteStore

    if isinstance(store, SQLiteStore):
        with store._lock:
            rows = store.con.execute(
                "SELECT l.liked, l.embedding, l.source, l.created_at FROM labels l "
                f"WHERE {_LABEL_NOT_RETRACTED}"
            ).fetchall()
        return [
            (bool(liked), json.loads(emb), source, None if at is None else float(at))
            for liked, emb, source, at in rows
        ]
    if isinstance(store, BigQueryStore):
        rows = store.client.query(
            "SELECT l.liked, l.embedding, l.source, UNIX_MICROS(l.created_at) AS t_us "
            f"FROM `{store._tid('labels')}` l WHERE {store._label_not_retracted()}"
        ).result()
        return [
            (
                bool(r["liked"]),
                list(r["embedding"]),
                r.get("source"),
                None if r["t_us"] is None else int(r["t_us"]) / 1e6,
            )
            for r in rows
        ]
    return None


def label_times(store, v1_rows: Sequence) -> tuple[list[float] | None, str]:
    """created_at per loaded v1 row (same order), or ``(None, why not)``."""
    try:
        stamped = _stamped_label_rows(store)
    except Exception as exc:  # noqa: BLE001 - the time-forward section is a diagnostic
        return None, f"label timestamps could not be read ({type(exc).__name__}: {exc})"
    if stamped is None:
        return None, "this store cannot report label timestamps"
    pool: dict[tuple, list[float]] = defaultdict(list)
    for liked, embedding, source, created_at in stamped:
        if created_at is None or not math.isfinite(created_at):
            return None, "a label has no created_at"
        pool[_row_key(liked, embedding, source)].append(created_at)
    for bucket in pool.values():
        bucket.sort(
            reverse=True
        )  # pop() hands identical rows their timestamps oldest first
    out: list[float] = []
    for liked, embedding, source in v1_rows:
        bucket = pool.get(_row_key(liked, embedding, source))
        if not bucket:
            return None, "label timestamps do not match the loaded labels"
        out.append(bucket.pop())
    return out, ""


# --------------------------------------------------------------------------- CLI


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="compare_feature_sets",
        description="Pre-registered comparison of face_crop_v1 and face_crop_whole_photo_v1.",
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--seeds",
        type=_positive_int,
        default=DEFAULT_SEEDS,
        help=f"shuffled CV repetitions (pre-registered: {DEFAULT_SEEDS})",
    )
    ap.add_argument(
        "--orders",
        type=_positive_int,
        default=DEFAULT_ORDERS,
        help=f"row permutations for the nested evaluation (pre-registered: {DEFAULT_ORDERS})",
    )
    ap.add_argument(
        "--jobs",
        type=_positive_int,
        default=1,
        help="threads for the nested evaluation (default 1; results do not depend on it)",
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    from operation_love import config as cfg_mod
    from operation_love.ranker import make_store

    cfg = cfg_mod.load(args.config)
    recall_target = cfg.ranker.like_recall_target
    if recall_target is None:
        print(
            "Cannot compare: ranker.like_recall_target is not set, and the pre-registered "
            "rule is defined at the configured recall target."
        )
        return EXIT_REFUSED
    store = make_store(cfg, ensure=False)  # read-only: creates nothing
    try:
        v1_rows, wp_rows = load_rows(store)
        times, time_note = label_times(store, v1_rows)
    except ComparisonRefused as exc:
        print(f"Cannot compare: {exc}")
        return EXIT_REFUSED
    finally:
        store.close()
    try:
        report = compare(
            v1_rows,
            wp_rows,
            seeds=args.seeds,
            orders=args.orders,
            recall_target=recall_target,
            times=times,
            time_note=time_note,
            jobs=args.jobs,
        )
    except ComparisonRefused as exc:
        print(f"Cannot compare: {exc}")
        return EXIT_REFUSED
    print(format_report(report))
    return EXIT_SWITCH if report["verdict"]["switch"] else EXIT_KEEP


if __name__ == "__main__":
    sys.exit(main())
