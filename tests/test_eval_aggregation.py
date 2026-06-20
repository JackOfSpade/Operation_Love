"""Identity grouping + result shape for leakage-free evaluation. Needs sklearn."""
from operation_love.ranker.evaluate import evaluate, format_report, identity_groups


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
