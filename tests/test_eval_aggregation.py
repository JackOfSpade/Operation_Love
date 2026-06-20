"""Identity grouping for leakage-free evaluation. Needs sklearn (ml extra)."""
import contextlib
import io

from tools.eval_aggregation import evaluate, identity_groups


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


def test_evaluate_too_few_labels_is_graceful():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluate([(True, [0.0] * 1280), (False, [1.0] * 1280)])   # 2 labels -> graceful, no crash
    assert "Need >=10 labels" in buf.getvalue()


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
