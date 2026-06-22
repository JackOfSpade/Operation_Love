"""Identity grouping + result shape for leakage-free evaluation. Needs sklearn."""
import importlib

from operation_love.ranker.evaluate import (
    evaluate,
    format_report,
    identity_groups,
    marginal_return_summary,
)


eval_mod = importlib.import_module("operation_love.ranker.evaluate")


def test_identity_groups_merges_same_face_separates_others():
    # Identity A appears 3x (near [1,0]), identity B twice (near [0,1]).
    faces = [[1.0, 0.0], [0.99, 0.02], [0.98, 0.0], [0.0, 1.0], [0.03, 0.99]]
    g = identity_groups(faces, eps=0.5)
    assert g[0] == g[1] == g[2]          # all three A photos share one identity group
    assert g[3] == g[4]                  # both B photos share one group
    assert g[0] != g[3]                  # A and B are different identities


def test_identity_groups_singletons_get_own_group():
    faces = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]   # three distinct, orthogonal/opposite
    assert len(set(identity_groups(faces, eps=0.5))) == 3


def test_evaluate_too_few_labels_returns_status_not_metrics():
    r = evaluate([(True, [0.0] * 1280), (False, [1.0] * 1280)])   # 2 labels
    assert r["status"] == "insufficient_data"
    assert r["roc_auc"] is None and r["labels"] == 2
    assert "Need" in format_report(r)                # renders the message, not numbers


def test_evaluate_never_raises_on_malformed_embeddings():
    # ragged embeddings (mixed lengths), enough labels with both classes -> must NOT raise.
    samples = [(i % 2 == 0, [0.0] * (1280 if i % 3 else 8)) for i in range(12)]
    r = evaluate(samples)
    assert r["status"] == "error"                    # handled gracefully, no traceback


def _identity_samples(n):
    samples = []
    for i in range(n):
        emb = [0.0] * 1280
        emb[i % 512] = 1.0                           # one distinct identity per row
        emb[512] = (i % 17) / 17.0                   # harmless ranker feature variation
        samples.append((i % 4 == 0, emb))
    return samples


def _fake_saturating_evaluate(samples, n_splits=5, eps=0.5):
    n = len(samples)
    likes = sum(1 for liked, _ in samples if liked)
    passes = n - likes
    if n < 10 or not likes or not passes:
        return {"status": "insufficient_data", "labels": n, "likes": likes, "passes": passes,
                "identities": n, "folds": 0, "roc_auc": None, "pr_auc": None,
                "brier": None, "base_rate": (likes / n) if n else 0.0,
                "message": "too small"}
    std = getattr(_fake_saturating_evaluate, "std", 0.001)
    pr = 0.72 - (3.0 / n)
    return {"status": "ok", "labels": n, "likes": likes, "passes": passes,
            "identities": n, "folds": min(n_splits, likes, passes),
            "roc_auc": [0.70, std], "pr_auc": [pr, std], "brier": [0.20, 0.01],
            "base_rate": likes / n, "message": "fake grouped CV"}


def _fake_flat_evaluate(samples, n_splits=5, eps=0.5):
    # Saturated learning curve whose top CV points tick slightly DOWN with n (the
    # signature of a plateau under fold noise): the reciprocal fit slope is non-positive
    # (b<=0) and the finite-difference secant is negative, so no positive gain resolves.
    n = len(samples)
    likes = sum(1 for liked, _ in samples if liked)
    passes = n - likes
    if n < 10 or not likes or not passes:
        return {"status": "insufficient_data", "labels": n, "likes": likes, "passes": passes,
                "identities": n, "folds": 0, "roc_auc": None, "pr_auc": None,
                "brier": None, "base_rate": (likes / n) if n else 0.0,
                "message": "too small"}
    pr = 0.71 - 0.0002 * n
    return {"status": "ok", "labels": n, "likes": likes, "passes": passes,
            "identities": n, "folds": min(n_splits, likes, passes),
            "roc_auc": [0.70, 0.01], "pr_auc": [pr, 0.01], "brier": [0.20, 0.01],
            "base_rate": likes / n, "message": "fake saturated CV"}


def test_marginal_return_decreases_as_labels_grow(monkeypatch):
    monkeypatch.setattr(eval_mod, "evaluate", _fake_saturating_evaluate)
    small = marginal_return_summary(_identity_samples(50), batch=20)
    large = marginal_return_summary(_identity_samples(140), batch=20)

    assert small["status"] == "ok" and large["status"] == "ok"
    assert small["marginal_return"] > large["marginal_return"] > 0


def test_marginal_return_small_n_is_too_early():
    r = marginal_return_summary(_identity_samples(12))
    assert r["status"] == "too_early"
    assert r["marginal_return"] is None


def test_marginal_return_within_noise_tracks_cv_std(monkeypatch):
    monkeypatch.setattr(eval_mod, "evaluate", _fake_saturating_evaluate)

    _fake_saturating_evaluate.std = 0.001
    clear = marginal_return_summary(_identity_samples(80), batch=20)
    assert clear["status"] == "ok"
    assert clear["within_noise"] is False

    _fake_saturating_evaluate.std = 0.05
    noisy = marginal_return_summary(_identity_samples(80), batch=20)
    assert noisy["status"] == "ok"
    assert noisy["within_noise"] is True

    _fake_saturating_evaluate.std = 0.001


def test_marginal_return_never_raises_on_edge_inputs():
    cases = [
        None,
        [],
        ["not a sample"] * 30,
        [(i % 2 == 0, [0.0] * (1280 if i % 3 else 8)) for i in range(30)],
    ]
    for samples in cases:
        r = marginal_return_summary(samples)
        assert isinstance(r, dict)
        assert "status" in r


def test_format_report_includes_diminishing_returns_line():
    r = {
        "status": "ok", "labels": 80, "likes": 20, "passes": 60,
        "identities": 80, "folds": 5, "roc_auc": [0.70, 0.02],
        "pr_auc": [0.42, 0.03], "brier": [0.20, 0.01], "base_rate": 0.25,
        "marginal_return": {
            "status": "ok", "batch": 20, "marginal_return": 0.0032,
            "within_noise": False, "confidence": "med", "message": "ok",
        },
    }
    out = format_report(r)
    assert "Diminishing returns" in out
    assert "0.0032 PR-AUC / +20 labels" in out


def test_format_report_shows_dash_when_no_estimate():
    r = {
        "status": "error",
        "message": "evaluation failed",
        "marginal_return": {
            "status": "error", "batch": 20, "marginal_return": None,
            "within_noise": None, "confidence": "low", "message": "bad input",
        },
    }
    out = format_report(r)
    assert "Diminishing returns —" in out
    assert out.startswith("Evaluation failed")
    assert "too early" not in out and "keep seeding" not in out


def test_format_marginal_value_keeps_enough_decimals_to_stay_nonzero():
    fmt = eval_mod._format_marginal_value
    assert fmt(0.014) == "0.014"
    assert fmt(0.0021) == "0.0021"
    assert fmt(0.00038) == "0.00038"
    assert fmt(0.000061) == "0.000061"
    # never collapses to zero, no matter how small the value gets
    for v in (1e-3, 1e-5, 1e-7):
        s = fmt(v)
        assert s is not None and float(s) > 0


def test_marginal_return_plateau_stays_a_positive_nonzero_value(monkeypatch):
    # A flat/declining curve must NOT die to "too early" or zero: report a tiny positive
    # magnitude (status ok) so the single indicator stays live and keeps shrinking.
    monkeypatch.setattr(eval_mod, "evaluate", _fake_flat_evaluate)
    r = marginal_return_summary(_identity_samples(80), batch=20)
    assert r["status"] == "ok"
    assert r["marginal_return"] > 0
    line = eval_mod._format_marginal_return(r)
    assert "Diminishing returns" in line
    assert "too early" not in line and "keep seeding" not in line


def test_marginal_return_hard_eval_failure_is_error(monkeypatch):
    # When evaluate() fails for a non-quantity reason (sklearn missing / malformed data),
    # more labels won't help — show the dash, never "too early"/"keep seeding".
    def _no_sklearn(samples, n_splits=5, eps=0.5):
        return {"status": "no_sklearn", "labels": len(samples), "likes": 0, "passes": 0,
                "identities": None, "folds": 0, "roc_auc": None, "pr_auc": None,
                "brier": None, "base_rate": 0.0, "message": "scikit-learn unavailable"}
    monkeypatch.setattr(eval_mod, "evaluate", _no_sklearn)
    r = marginal_return_summary(_identity_samples(40), batch=20)
    assert r["status"] == "error"
    assert eval_mod._format_marginal_return(r) == "Diminishing returns —"


def test_marginal_return_post_ok_group_contradiction_is_error():
    samples = [(i % 2 == 0, [1.0] + [0.0] * 1279) for i in range(30)]
    stale_ok = {"status": "ok", "pr_auc": [0.50, 0.05]}

    r = marginal_return_summary(samples, eval_result=stale_ok)
    assert r["status"] == "error"
    assert eval_mod._format_marginal_return(r) == "Diminishing returns —"


def test_quality_trajectory_walks_prefixes_every_step(monkeypatch):
    # Reconstructs the metric history at chronological prefixes, every `step` labels.
    monkeypatch.setattr(eval_mod, "evaluate", _fake_saturating_evaluate)
    traj = eval_mod.quality_trajectory(_identity_samples(53), step=5)
    assert [p["labels"] for p in traj][:3] == [10, 15, 20]   # starts at min_labels (10), every 5
    assert traj[-1]["labels"] == 53                          # full set is always the final point
    assert traj[0]["pr_auc"] < traj[-1]["pr_auc"]            # saturating curve rises with n
    assert all(p.get("roc_auc") is not None and p.get("base_rate") is not None for p in traj)


def test_quality_trajectory_never_raises_on_edge_inputs():
    for samples in (None, [], ["junk"] * 8):
        assert eval_mod.quality_trajectory(samples, step=5) == []   # too small / malformed -> empty


def test_quality_trajectory_caps_point_count_as_labels_grow(monkeypatch):
    # Cost is one full grouped CV per point; the step widens so points stay bounded at scale.
    monkeypatch.setattr(eval_mod, "evaluate", _fake_saturating_evaluate)
    big = eval_mod.quality_trajectory(_identity_samples(1000), step=5, max_points=40)
    assert len(big) <= 41                                  # ~max_points, not 200 (= 1000/5)
    assert big[-1]["labels"] == 1000                       # still ends on the full set
    # small N keeps the fine step-5 granularity (cap doesn't kick in)
    assert [p["labels"] for p in eval_mod.quality_trajectory(_identity_samples(53), step=5)][:3] == [10, 15, 20]


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    sys.exit(1 if failed else 0)
