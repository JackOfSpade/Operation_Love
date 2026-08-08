"""Vision pure-logic tests — aggregation + quality gating (no torch/pyiqa)."""
import math
import types

from operation_love.perception.capture import Profile
from operation_love.vision.embed import (
    Embedder, _is_onnx_provider_failure, _select_onnx_providers, aggregate, concat,
    dedup_by_cosine, gem_pool, l2_normalize, square_crop_around_bbox,
)
from operation_love.vision.quality import QualityFilter


def test_square_crop_around_bbox_is_square_and_in_bounds():
    left, top, right, bottom = square_crop_around_bbox((1080, 2400), [500, 1000, 600, 1200], expand=3.0)
    assert 0 <= left < right <= 1080 and 0 <= top < bottom <= 2400
    assert abs((right - left) - (bottom - top)) <= 1            # square


def test_square_crop_clamps_to_image_at_corner():
    left, top, right, bottom = square_crop_around_bbox((1080, 2400), [0, 0, 120, 120], expand=6.0)
    assert left >= 0 and top >= 0 and right <= 1080 and bottom <= 2400


def test_square_crop_side_never_exceeds_image():
    left, top, right, bottom = square_crop_around_bbox((1080, 2400), [400, 1100, 700, 1400], expand=99.0)
    assert (right - left) <= 1080 and (bottom - top) <= 2400    # clamped to min dimension


COREML_RUNTIME_ERROR = (
    "[ONNXRuntimeError] : 1 : FAIL : CoreMLExecutionProvider CoreML static output "
    "shape ({1,1,1,128,1}) and inferred shape ({3200,1}) have different ranks."
)


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


def test_provider_failure_detector_matches_coreml_onnxruntime_failures():
    assert _is_onnx_provider_failure(COREML_RUNTIME_ERROR)
    assert _is_onnx_provider_failure(RuntimeError("[ONNXRuntimeError] : 1 : FAIL : failed"))
    assert _is_onnx_provider_failure("status: fail while running provider")
    assert not _is_onnx_provider_failure(ValueError("PIL cannot identify image file"))


class _FakeFace:
    bbox = [0.0, 0.0, 10.0, 10.0]
    embedding = [3.0, 4.0]


class _FakeArc:
    def __init__(self, providers):
        self.providers = list(providers)
        self.calls = 0

    def get(self, _img):
        self.calls += 1
        if self.providers and self.providers[0] == "CoreMLExecutionProvider":
            raise RuntimeError(COREML_RUNTIME_ERROR)
        return [_FakeFace()]


def test_embed_profile_rebuilds_arcface_on_cpu_after_coreml_runtime_failure():
    embedder = Embedder()
    embedder._device = "mps"
    embedder._arc_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    embedder._arc_on_cpu = False
    embedder._arc = _FakeArc(embedder._arc_providers)
    rebuilds = []

    def fake_build_arc(self, providers):
        rebuilds.append(list(providers))
        return _FakeArc(providers)

    def fake_embed_image(self, _img):
        faces = self._arc.get(None)
        return list(faces[0].embedding), [0.6, 0.8]

    embedder._build_arc = types.MethodType(fake_build_arc, embedder)
    embedder._embed_image = types.MethodType(fake_embed_image, embedder)

    vec = embedder.embed_profile(Profile(photos=[b"fake"]))

    assert rebuilds == [["CPUExecutionProvider"]]
    assert embedder._arc_on_cpu is True
    assert embedder._arc_providers == ["CPUExecutionProvider"]
    assert vec is not None
    assert len(vec) == 4
    assert math.isclose(math.sqrt(sum(x * x for x in vec[:2])), 1.0)
    assert math.isclose(math.sqrt(sum(x * x for x in vec[2:])), 1.0)


def test_ensure_leaves_embedder_retryable_if_clip_load_fails(monkeypatch):
    # If CLIP's load raises (weight download / OOM), _ensure() must NOT leave a
    # half-initialized embedder: _arc (the init sentinel) stays None so the next call
    # retries cleanly, instead of short-circuiting on _arc-set/_clip-None and then
    # TypeError-ing every embed into a silent "no_face" for the rest of the run.
    import open_clip
    import pytest

    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("CLIP weight download failed")

    monkeypatch.setattr(open_clip, "create_model_and_transforms", boom)
    e = Embedder()
    with pytest.raises(RuntimeError):
        e._ensure()
    assert e._arc is None and e._clip is None        # nothing committed -> retry-able
    assert calls["n"] == 1


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


def test_faceless_photo_clip_vector_excluded_from_profile_pool(monkeypatch):
    """A photo with no detected face still returns a CLIP vector for the WHOLE screenshot
    (_embed_image's clip_src falls back to the raw image, not a person-centric crop, when
    no face is found). embed_profile must not fold that into the pooled CLIP vector -- it
    would dilute the style/vibe signal with UI chrome (see square_crop_around_bbox's
    docstring). Only the faced photo's crop should survive into the pool."""
    embedder = Embedder()
    monkeypatch.setattr(embedder, "_ensure", lambda: None)
    results = iter([
        ([1.0, 0.0], [1.0, 0.0]),  # photo 1: face found -> person-crop clip vec
        (None, [0.0, 1.0]),        # photo 2: no face -> whole-screenshot clip vec, must be dropped
    ])
    monkeypatch.setattr(embedder, "_embed_image", lambda img: next(results))

    vec = embedder.embed_profile(Profile(photos=[b"a", b"b"]))

    assert vec == [1.0, 0.0, 1.0, 0.0]   # clip half == photo 1's vec alone, never photo 2's


def test_embed_profile_flags_total_failure_distinctly_from_no_face(monkeypatch, capsys):
    """When every photo's embedding call raises, that's a broken embedder (operator
    problem), not a genuine no-face profile -- both currently return None, but the total
    failure must print a distinguishable, loud signal so it isn't silently mistaken for a
    normal run of faceless profiles."""
    embedder = Embedder()
    monkeypatch.setattr(embedder, "_ensure", lambda: None)

    def boom(_img):
        raise RuntimeError("simulated total embed failure")

    monkeypatch.setattr(embedder, "_embed_image", boom)

    result = embedder.embed_profile(Profile(photos=[b"a", b"b"]))

    assert result is None
    out = capsys.readouterr().out
    assert "WARNING" in out and "broken embedder" in out


def test_embed_profile_returns_none_on_nan_in_final_vector(monkeypatch):
    """A NaN/Inf in the final concatenated vector (e.g. from a degenerate l2_normalize)
    must return None (treated as no_face) rather than storing a corrupted embedding."""
    from operation_love.vision.embed import Embedder
    from operation_love.perception.capture import Profile

    embedder = Embedder()
    # Inject a degenerate face vector (all zeros -> l2_normalize returns zeros -> concat has zeros,
    # not NaN, but we test the guard by patching concat to return a NaN-containing vector).
    import operation_love.vision.embed as embed_mod
    monkeypatch.setattr(embed_mod, "concat", lambda *_: [0.1, float("nan"), 0.3])
    # Also patch _ensure so it doesn't try to import ML libs.
    monkeypatch.setattr(embedder, "_ensure", lambda: None)
    # Patch embed_profile to simulate one detected face and one CLIP vector.
    monkeypatch.setattr(embedder, "_embed_image", lambda img: ([0.1, 0.2], [0.3, 0.4]))

    profile = Profile(photos=[b"fake_photo"])
    result = embedder.embed_profile(profile)
    assert result is None, "expected None when final vector contains NaN"
