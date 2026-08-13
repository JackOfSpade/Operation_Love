"""Vision pure-logic tests — aggregation + quality gating (no torch/pyiqa)."""
import math
import types

from operation_love.perception.capture import Profile
from operation_love.vision.embed import (
    _COREML_PROVIDER_OPTIONS, Embedder, _is_onnx_provider_failure, _onnx_provider_options,
    _select_onnx_providers, _synthetic_probe_image, aggregate, concat, dedup_by_cosine,
    gem_pool, l2_normalize, square_crop_around_bbox,
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


def test_onnx_provider_options_are_positionally_aligned_with_providers():
    # onnxruntime matches provider_options to providers BY POSITION, so the list has to be
    # the same length -- a non-CoreML provider gets an empty dict, never an omitted slot.
    assert _onnx_provider_options(["CoreMLExecutionProvider", "CPUExecutionProvider"]) == [
        {"ModelFormat": "MLProgram"}, {}
    ]
    assert _onnx_provider_options(["CUDAExecutionProvider", "CPUExecutionProvider"]) == [{}, {}]
    assert _onnx_provider_options(["CPUExecutionProvider"]) == [{}]
    assert _onnx_provider_options([]) == []


def test_coreml_uses_mlprogram_not_the_default_neuralnetwork_format():
    """Guards a measured decision, not a preference. onnxruntime's default CoreML model
    format (NeuralNetwork) runs w600k_r50 in fp16 on the ANE: 2.0ms/call but only 0.9957
    cosine against the CPU embedding. MLProgram is 5.5ms/call at 0.99999994. Every stored
    training label was collected against CPU vectors, so dropping back to the default would
    silently shift the ranker's feature space to save 3.5ms per photo."""
    assert _COREML_PROVIDER_OPTIONS == {"ModelFormat": "MLProgram"}


class _FakeSession:
    def __init__(self):
        self.provider_calls = []

    def set_providers(self, providers):
        self.provider_calls.append(list(providers))


class _FakeModel:
    def __init__(self):
        self.session = _FakeSession()


class _FakeFaceAnalysis:
    """Records what _build_arc did, in order. Stands in for insightface's FaceAnalysis so
    this test needs neither the package, the ~350MB of buffalo_l weights, nor a GPU."""

    instances = []

    def __init__(self, name=None, providers=None, provider_options=None):
        self.name = name
        self.providers = list(providers or [])
        self.provider_options = provider_options
        self.models = {"detection": _FakeModel(), "recognition": _FakeModel()}
        self.prepared_ctx = None
        # Detector providers at the moment prepare() ran: the pin only survives because it
        # happens BEFORE prepare(), so the ordering is part of what's under test.
        self.detector_providers_at_prepare = None
        _FakeFaceAnalysis.instances.append(self)

    def prepare(self, ctx_id=None, **kwargs):
        self.prepared_ctx = ctx_id
        self.detector_providers_at_prepare = list(self.models["detection"].session.provider_calls)


def _install_fake_insightface(monkeypatch):
    """Inject fake `insightface` / `insightface.app` modules via sys.modules (the same
    trick tests/test_concurrency.py uses for open_clip/onnxruntime) so the REAL
    Embedder._build_arc runs unmodified against a stand-in FaceAnalysis."""
    import sys

    _FakeFaceAnalysis.instances = []
    app = types.ModuleType("insightface.app")
    app.FaceAnalysis = _FakeFaceAnalysis
    pkg = types.ModuleType("insightface")
    pkg.app = app
    monkeypatch.setitem(sys.modules, "insightface", pkg)
    monkeypatch.setitem(sys.modules, "insightface.app", app)


def test_build_arc_pins_the_detector_to_cpu_when_coreml_is_selected(monkeypatch):
    """buffalo_l's det_10g declares 640x640-shaped outputs but insightface 1.0.1 runs it
    multi-scale (128x128 AND 640x640 per call), and the CoreML EP hard-asserts on the
    resulting declared-vs-actual output shape mismatch -- the rank error that used to fail
    every run. The detector must therefore be moved to CPU while the rest of buffalo_l
    stays on CoreML, and it must happen BEFORE prepare()."""
    _install_fake_insightface(monkeypatch)
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    arc = Embedder()._build_arc(providers)

    assert arc.providers == providers                       # CoreML still attempted, not dropped
    assert arc.provider_options == [{"ModelFormat": "MLProgram"}, {}]
    assert arc.models["detection"].session.provider_calls == [["CPUExecutionProvider"]]
    assert arc.models["recognition"].session.provider_calls == []   # recognition stays on CoreML
    assert arc.detector_providers_at_prepare == [["CPUExecutionProvider"]]   # pinned pre-prepare
    assert arc.prepared_ctx == 0


def test_build_arc_on_cpu_only_touches_no_provider_and_uses_cpu_ctx(monkeypatch):
    # Nothing to work around when CoreML was never selected: no per-model provider surgery,
    # no CoreML options, and ctx_id=-1 (insightface's own "everything on CPU" signal).
    _install_fake_insightface(monkeypatch)

    arc = Embedder()._build_arc(["CPUExecutionProvider"])

    assert arc.provider_options == [{}]
    assert arc.models["detection"].session.provider_calls == []
    assert arc.prepared_ctx == -1


def test_build_arc_on_cuda_does_not_pin_the_detector(monkeypatch):
    # The declared-shape mismatch is a CoreML-EP assert; the CUDA EP handles the dynamic
    # detector input fine, so it must not inherit a macOS-specific workaround.
    _install_fake_insightface(monkeypatch)

    arc = Embedder()._build_arc(["CUDAExecutionProvider", "CPUExecutionProvider"])

    assert arc.provider_options == [{}, {}]
    assert arc.models["detection"].session.provider_calls == []
    assert arc.prepared_ctx == 0


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


def test_embedder_warmup_reraises_init_failure(monkeypatch):
    """warmup() is meant to be called once, eagerly, on the main thread before workers
    start (see its docstring), precisely so a broken embedder is caught before any worker
    thread ever comes to depend on it. Old contract: warmup() caught any _ensure() failure
    and just printed "will retry per-profile" -- but that "retry" runs unguarded inside a
    worker thread, only after supervisor.run() has already finished the rest of startup,
    taken the Android device lock, and opened a live session, so the failure actually
    surfaced after a real profile had already been captured off the phone. warmup() must
    now re-raise instead, matching QualityFilter.warmup()'s fail-loud contract, so the
    supervisor aborts cleanly during startup."""
    import pytest

    def boom():
        raise RuntimeError("onnxruntime install is corrupt")

    e = Embedder()
    monkeypatch.setattr(e, "_ensure", boom)

    with pytest.raises(RuntimeError, match="onnxruntime install is corrupt"):
        e.warmup()


def test_quality_gating_with_injected_scorer():
    scores = {b"good": 0.8, b"bad": 0.1}
    qf = QualityFilter(enabled=True, min_score=0.3, scorer=lambda b: scores[b])
    assert qf.keep(b"good") is True
    assert qf.keep(b"bad") is False
    assert qf.filter([b"good", b"bad", b"good"]) == [b"good", b"good"]


def test_quality_disabled_keeps_all():
    qf = QualityFilter(enabled=False, min_score=0.9, scorer=lambda b: 0.0)
    assert qf.filter([b"a", b"b"]) == [b"a", b"b"]


def test_quality_fails_loud_on_scorer_error():
    """Old contract: keep() caught scorer exceptions and returned True, i.e. "fail open --
    never drop a photo because scoring broke". That is exactly backwards for a pipeline
    that trains on the photos it keeps: a scorer that has started raising isn't gracefully
    degrading the quality bar, it's OFF, and fail-open means every junk/blurry photo from
    then on silently passes the filter and corrupts the training labels the run exists to
    produce -- with no visible signal that quality gating stopped happening at all.
    keep() (and filter(), which calls it per photo) now deliberately let the scorer's
    exception propagate instead, so the worker's halt_on_error machinery stops the run
    immediately rather than continuing to silently mislabel photos through a broken
    filter."""
    import pytest

    def boom(_):
        raise ValueError("scorer failed")

    qf = QualityFilter(enabled=True, min_score=0.3, scorer=boom)

    with pytest.raises(ValueError, match="scorer failed"):
        qf.keep(b"x")

    with pytest.raises(ValueError, match="scorer failed"):
        qf.filter([b"good", b"x"])


def test_quality_warmup_reraises_init_failure():
    """warmup() is meant to be called once, eagerly, on the main thread before workers
    start (see its docstring in quality.py), precisely so a broken quality model is caught
    before any worker thread ever comes to depend on it. _ensure() latches an init failure
    onto self._init_error so it isn't retried on every single photo, but warmup() must
    still re-raise that failure so the supervisor aborts the run cleanly instead of
    silently degrading to no filtering."""
    import pytest

    qf = QualityFilter(enabled=True, min_score=0.3)
    qf._init_error = RuntimeError("model load failed")

    with pytest.raises(RuntimeError, match="model load failed"):
        qf.warmup()


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


def test_synthetic_probe_image_is_a_valid_deterministic_in_memory_image():
    """_probe_inference()'s warmup probe must never touch disk or the network, and must
    be reproducible (same content every run) so a probe failure is easy to reason about
    instead of depending on random content that happened to trip a provider bug."""
    import io

    from PIL import Image

    b1 = _synthetic_probe_image()
    b2 = _synthetic_probe_image()
    assert isinstance(b1, bytes) and len(b1) > 0
    assert b1 == b2                                       # fixed seed -> reproducible probe
    img = Image.open(io.BytesIO(b1)).convert("RGB")
    assert img.size == (128, 128)


def test_warmup_probe_catches_coreml_failure_invisible_to_build_only_warmup(monkeypatch, capsys):
    """The real bug this project hit: CoreMLExecutionProvider's ort.InferenceSession(...)
    for buffalo_l BUILDS without error -- the plain _ensure() gate that existed before
    this fix saw a clean success -- but the compiled graph's static-shape incompatibility
    only threw on the first real .get() call, which used to mean the operator's first
    live profile. warmup() must now catch this itself, via a synthetic probe inference
    routed through embed_profile()'s existing CPU-fallback machinery, landing the
    embedder on CPU before any worker ever touches it -- and it must NOT raise, since a
    provider hiccup the existing fallback can recover from is not a fatal startup error
    (only a genuine _ensure() failure is)."""
    embedder = Embedder()
    monkeypatch.setattr(embedder, "_ensure", lambda: None)
    embedder._arc_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    embedder._arc_on_cpu = False
    rebuilds = []

    def fake_build_arc(self, providers):
        rebuilds.append(list(providers))
        return _FakeArc(providers)

    def fake_embed_image(self, _img):
        # BUILD already "succeeded" (that's the whole bug); only real INFERENCE fails,
        # and only while still on the CoreML provider.
        if not self._arc_on_cpu:
            raise RuntimeError(COREML_RUNTIME_ERROR)
        return [1.0, 0.0], [0.5, 0.5]

    embedder._build_arc = types.MethodType(fake_build_arc, embedder)
    embedder._embed_image = types.MethodType(fake_embed_image, embedder)

    embedder.warmup()  # must not raise

    assert rebuilds == [["CPUExecutionProvider"]]
    assert embedder._arc_on_cpu is True
    out = capsys.readouterr().out
    assert "CoreML/ONNX provider failed; retrying this profile on CPU" in out


def test_warmup_probe_does_not_hard_fail_on_non_provider_shaped_error(monkeypatch):
    """A synthetic-probe failure that ISN'T provider-shaped (e.g. some unrelated bug in
    the tiny generated image itself) must not turn into a hard startup failure -- only a
    genuine _ensure() failure gates startup. embed_profile() already swallows any single
    photo's embedding error (counts it, prints it, never raises), so this falls out of
    reusing that path rather than needing a separate try/except in warmup()."""
    embedder = Embedder()
    monkeypatch.setattr(embedder, "_ensure", lambda: None)

    def fake_embed_image(_img):
        raise ValueError("PIL cannot identify image file")   # deliberately not provider-shaped

    monkeypatch.setattr(embedder, "_embed_image", fake_embed_image)

    embedder.warmup()  # must not raise despite every probe photo erroring


def test_embed_profile_falls_back_to_cpu_on_partial_not_total_coreml_failure():
    """Old contract: the CPU fallback only fired when EVERY photo errored
    (errors == len(photos)). A provider-shaped failure is deterministic per call in
    reality (see the module's motivating case), so errors < len(photos) here just means
    the run got lucky about which photos ran before the exception, not that the provider
    is partly healthy. 2-of-3 photos erroring on CoreML must still retry the WHOLE
    profile on CPU rather than silently pooling from the 1 survivor."""
    embedder = Embedder()
    embedder._device = "mps"
    embedder._arc_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    embedder._arc_on_cpu = False
    embedder._arc = _FakeArc(embedder._arc_providers)
    rebuilds = []
    calls = {"n": 0}

    def fake_build_arc(self, providers):
        rebuilds.append(list(providers))
        return _FakeArc(providers)

    def fake_embed_image(self, _img):
        calls["n"] += 1
        if self._arc_on_cpu:
            return [1.0, 0.0], [0.6, 0.8]
        if calls["n"] in (1, 2):                  # 2 of 3 photos fail on CoreML this pass
            raise RuntimeError(COREML_RUNTIME_ERROR)
        return [9.0, 0.0], [1.0, 0.0]              # the 1 "survivor" a pre-fix run would pool alone

    embedder._build_arc = types.MethodType(fake_build_arc, embedder)
    embedder._embed_image = types.MethodType(fake_embed_image, embedder)

    vec = embedder.embed_profile(Profile(photos=[b"a", b"b", b"c"]))

    assert rebuilds == [["CPUExecutionProvider"]]   # fallback DID trigger on a 2-of-3 failure
    assert embedder._arc_on_cpu is True
    assert vec is not None


def test_embed_profile_warns_on_partial_failure_with_no_fallback_left(monkeypatch, capsys):
    """Even when there's nowhere left to fall back to (already on CPU) -- or the errors
    aren't provider-shaped at all -- a partially-embedded profile must never look
    indistinguishable from a clean one: whenever ANY photo errors, a clear WARNING states
    how many, since the pooled embedding is quietly built from fewer photos than the
    profile actually has."""
    embedder = Embedder()
    monkeypatch.setattr(embedder, "_ensure", lambda: None)
    embedder._arc_on_cpu = True  # already on CPU -- nowhere left to fall back to

    results = iter([([1.0, 0.0], [0.6, 0.8])])   # only the 1st of 3 photos succeeds

    def fake_embed_image(_img):
        try:
            return next(results)
        except StopIteration:
            raise RuntimeError("corrupt JPEG")

    monkeypatch.setattr(embedder, "_embed_image", fake_embed_image)

    vec = embedder.embed_profile(Profile(photos=[b"a", b"b", b"c"]))

    out = capsys.readouterr().out
    assert "WARNING: 2/3 photo(s) failed to embed in this profile" in out
    assert vec is not None    # fail loud (warn), not fail closed (the 1 survivor still pools)


def test_embed_profile_summary_carries_failed_first_pass_after_cpu_recovery(capsys):
    """Old contract: `errors` was reset at the top of each retry iteration, so once the
    CPU retry succeeded the per-profile summary read a perfectly clean
    "N photo(s) -> N with a face" with zero hint that the first pass had failed
    outright -- from a line that exists precisely so 'no_face is never a silent
    mystery'. The summary must carry the failed first pass's count and provider name
    forward across the retry."""
    embedder = Embedder()
    embedder._arc_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    embedder._arc_on_cpu = False
    embedder._arc = _FakeArc(embedder._arc_providers)

    def fake_build_arc(self, providers):
        return _FakeArc(providers)

    def fake_embed_image(self, _img):
        faces = self._arc.get(None)
        return list(faces[0].embedding), [0.6, 0.8]

    embedder._build_arc = types.MethodType(fake_build_arc, embedder)
    embedder._embed_image = types.MethodType(fake_embed_image, embedder)

    photos = [f"photo{i}".encode() for i in range(8)]
    embedder.embed_profile(Profile(photos=photos))

    out = capsys.readouterr().out
    assert ("Profile: 8 photo(s) -> 8 with a face "
            "(8 errored on CoreMLExecutionProvider, re-run on CPU)") in out
