"""Vision pure-logic tests — aggregation + quality gating (no torch/pyiqa)."""
from operation_love.vision.embed import aggregate, concat
from operation_love.vision.quality import QualityFilter


def test_aggregate_mean():
    assert aggregate([[1.0, 2.0], [3.0, 4.0]]) == [2.0, 3.0]
    assert aggregate([]) == []


def test_concat():
    assert concat([1.0], [2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_quality_gating_with_injected_scorer():
    scores = {b"good": 0.8, b"bad": 0.1}
    qf = QualityFilter(enabled=True, min_score=0.3, scorer=lambda b: scores[b])
    assert qf.keep(b"good") is True
    assert qf.keep(b"bad") is False
    assert qf.filter([b"good", b"bad", b"good"]) == [b"good", b"good"]


def test_quality_disabled_keeps_all():
    qf = QualityFilter(enabled=False, min_score=0.9, scorer=lambda b: 0.0)
    assert qf.filter([b"a", b"b"]) == [b"a", b"b"]


def test_quality_never_drops_on_scorer_error():
    def boom(_):
        raise ValueError("scorer failed")
    qf = QualityFilter(enabled=True, min_score=0.3, scorer=boom)
    assert qf.keep(b"x") is True   # fail-open: never drop a photo because scoring broke


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
