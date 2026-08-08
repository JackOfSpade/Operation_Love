"""Identity grouping + result shape for leakage-free evaluation. Needs sklearn."""
import importlib

from operation_love.ranker.evaluate import (
    evaluate,
    format_report,
    identity_groups,
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
    score = 0.72 - (3.0 / n)
    return {"status": "ok", "labels": n, "likes": likes, "passes": passes,
            "identities": n, "folds": min(n_splits, likes, passes),
            "roc_auc": [score, 0.01], "pr_auc": [score, 0.01], "brier": [0.20, 0.01],
            "base_rate": likes / n, "message": "fake grouped CV"}


def test_format_report_shows_accuracy_brier_and_base_rate():
    r = {
        "status": "ok", "labels": 80, "likes": 20, "passes": 60,
        "identities": 80, "folds": 5, "roc_auc": [0.83, 0.04],
        "pr_auc": [0.42, 0.03], "brier": [0.20, 0.01], "base_rate": 0.25,
    }
    out = format_report(r)
    assert "Accuracy: 83.000% +/- 4.000%" in out
    assert "ranks a like above a pass" in out
    assert "Brier: 0.200 +/- 0.010" in out
    assert "Base rate: 25.0% likes" in out
    assert "PR-AUC" not in out and "Diminishing returns" not in out


def test_format_report_nonok_returns_message_only():
    out = format_report({"status": "error", "message": "evaluation failed"})
    assert out.startswith("Evaluation failed")
    assert "Accuracy" not in out and "Diminishing returns" not in out


def test_quality_trajectory_walks_prefixes_every_step(monkeypatch):
    # Reconstructs the metric history at chronological prefixes, every `step` labels.
    monkeypatch.setattr(eval_mod, "evaluate", _fake_saturating_evaluate)
    traj = eval_mod.quality_trajectory(_identity_samples(53), step=5)
    assert [p["labels"] for p in traj][:3] == [10, 15, 20]   # starts at min_labels (10), every 5
    assert traj[-1]["labels"] == 53                          # full set is always the final point
    assert traj[0]["roc_auc"] < traj[-1]["roc_auc"]          # visible accuracy curve rises with n
    assert all(p.get("roc_std") is not None and p.get("base_rate") is not None for p in traj)


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


def _grouped_cv_samples(n_identities=10, per_identity=4):
    # Real evaluate() 'ok' path: first 512 dims = L2-normed ArcFace face template (identity),
    # a non-face dim linearly separates likes from passes. Same identity shares a near-identical
    # normalized face prefix (one-hot per person => cosine dist 0 within, 1 across); deterministic.
    samples = []
    for ident in range(n_identities):
        face = [0.0] * 512
        face[ident] = 1.0                            # unit-norm face prefix unique to this identity
        for k in range(per_identity):
            emb = list(face) + [0.0] * 768
            liked = (k % 2 == 0)
            emb[512] = 1.0 if liked else -1.0        # non-face dim separates like/pass
            samples.append((liked, emb))
    return samples


def test_evaluate_real_grouped_cv_ok_path():
    # Drive the REAL sklearn StratifiedGroupKFold path (no monkeypatch): 40 rows / 10 identities.
    samples = _grouped_cv_samples(n_identities=10, per_identity=4)
    r = evaluate(samples)
    assert r["status"] == "ok"
    assert r["identities"] == 10                      # exactly the number of distinct faces built
    assert r["folds"] >= 2
    for key in ("roc_auc", "pr_auc", "brier"):
        mean, std = r[key]                            # each metric is a [mean, std] 2-list
        assert 0.0 <= mean <= 1.0 and 0.0 <= std <= 1.0


def test_evaluate_real_grouped_cv_too_few_identities():
    # Same face prefix on every row -> one identity -> can't split groups across folds.
    samples = []
    for i in range(12):
        emb = [0.0] * 1280
        emb[0] = 1.0                                  # identical face prefix => single identity group
        emb[512] = 1.0 if i % 2 == 0 else -1.0
        samples.append((i % 2 == 0, emb))
    r = evaluate(samples)
    assert r["status"] == "insufficient_groups"
    assert r["identities"] == 1
