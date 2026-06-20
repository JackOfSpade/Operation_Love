"""Vision pure-logic tests — aggregation + quality gating (no torch/pyiqa)."""
import math

from operation_love.vision.embed import (
    _select_onnx_providers, aggregate, concat, dedup_by_cosine, gem_pool, l2_normalize,
)
from operation_love.vision.quality import QualityFilter


def test_aggregate_mean():
    assert aggregate([[1.0, 2.0], [3.0, 4.0]]) == [2.0, 3.0]
    assert aggregate([]) == []


def test_concat():
    assert concat([1.0], [2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_l2_normalize_unit_length_and_zero_safe():
    out = l2_normalize([3.0, 4.0])
    assert math.isclose(out[0], 0.6) and math.isclose(out[1], 0.8)
    assert math.isclose(math.sqrt(sum(x * x for x in out)), 1.0)
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]      # no divide-by-zero


def test_gem_pool_single_is_identity_and_amplifies_salient():
    assert gem_pool([[0.2, -0.5, 0.9]]) == [0.2, -0.5, 0.9]      # K=1 -> identity
    # GeM(p=3) sits between mean and max, so a spike dominates more than the mean would.
    col = [0.1, 0.1, 0.9]
    g = gem_pool([[v] for v in col], p=3.0)[0]
    assert (sum(col) / 3) < g < max(col)


def test_gem_pool_preserves_sign():
    out = gem_pool([[-0.8], [-0.6]], p=3.0)
    assert out[0] < 0                               # dominant sign kept negative


def test_dedup_by_cosine_drops_near_duplicates():
    a = l2_normalize([1.0, 0.0])
    a2 = l2_normalize([0.99, 0.01])                # nearly identical to a -> dropped
    b = l2_normalize([0.0, 1.0])                   # orthogonal -> kept
    kept = dedup_by_cosine([a, a2, b], threshold=0.85)
    assert kept == [a, b]


def test_select_onnx_providers_prefers_coreml_on_mps():
    available = ["CoreMLExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"]
    assert _select_onnx_providers("mps", available) == [
        "CoreMLExecutionProvider", "CPUExecutionProvider"
    ]


def test_select_onnx_providers_prefers_cuda_when_available():
    available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert _select_onnx_providers("cuda", available) == [
        "CUDAExecutionProvider", "CPUExecutionProvider"
    ]


def test_select_onnx_providers_omits_unavailable_entries():
    available = ["AzureExecutionProvider", "CPUExecutionProvider"]
    assert _select_onnx_providers("cuda", available) == ["CPUExecutionProvider"]
    assert _select_onnx_providers("mps", available) == ["CPUExecutionProvider"]
    assert _select_onnx_providers("cpu", available) == ["CPUExecutionProvider"]


def test_select_onnx_providers_never_returns_empty():
    # Even if nothing preferred is available, fall back to CPU (onnxruntime
    # raises on an empty provider list).
    assert _select_onnx_providers("mps", []) == ["CPUExecutionProvider"]
    assert _select_onnx_providers("cuda", ["AzureExecutionProvider"]) == ["CPUExecutionProvider"]


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
