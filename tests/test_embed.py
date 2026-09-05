"""Embedder.embed_profile's CPU-fallback classification (no torch/onnxruntime).

Kept separate from tests/test_vision.py, which owns the pure aggregation/quality logic plus
the original provider-fallback tests: this file pins the ONE property those did not, that the
fallback decision is about whether ANY photo failed provider-shaped and not about which
exception happened to arrive first.
"""
import types

from operation_love.perception.capture import Profile
from operation_love.vision.embed import Embedder

# The real CoreMLExecutionProvider rank-mismatch message this machine produces (mirrors the
# constant in tests/test_vision.py; _is_onnx_provider_failure matches on its shape).
COREML_RUNTIME_ERROR = (
    "[ONNXRuntimeError] : 1 : FAIL : CoreMLExecutionProvider CoreML static output "
    "shape ({1,1,1,128,1}) and inferred shape ({3200,1}) have different ranks."
)
# What a truncated `adb screencap` PNG raises out of PIL: a real, routine, per-photo failure
# that says nothing whatever about the provider (_is_onnx_provider_failure rejects it).
CORRUPT_PHOTO_ERROR = "cannot identify image file"


def _embedder_on_coreml():
    embedder = Embedder()
    embedder._device = "mps"
    embedder._arc_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    embedder._arc_on_cpu = False
    embedder._arc = object()          # non-None: short-circuits _ensure()'s lazy model load
    return embedder


def _run_with_corrupt_photo_at(position, photo_count=3):
    """Embed a profile where exactly one photo is undecodable and every OTHER photo fails
    provider-shaped, with the corrupt one placed at `position`. Returns (rebuilds, embedder,
    vector)."""
    embedder = _embedder_on_coreml()
    rebuilds = []
    photos = [b"ok"] * photo_count
    photos[position] = b"corrupt"

    def fake_build_arc(self, providers):
        rebuilds.append(list(providers))
        return object()

    def fake_embed_image(self, img):
        if img == b"corrupt":
            # Stays broken on every provider -- it is the bytes that are bad, not the graph.
            raise ValueError(CORRUPT_PHOTO_ERROR)
        if not self._arc_on_cpu:
            raise RuntimeError(COREML_RUNTIME_ERROR)
        return [1.0, 0.0], [0.6, 0.8]

    embedder._build_arc = types.MethodType(fake_build_arc, embedder)
    embedder._embed_image = types.MethodType(fake_embed_image, embedder)
    return rebuilds, embedder, embedder.embed_profile(Profile(photos=photos))


def test_cpu_fallback_fires_when_a_corrupt_photo_precedes_the_provider_failure(capsys):
    """One undecodable photo ORDERED FIRST must not decide the provider question for the
    photos behind it. Pre-fix the classification read `first_error` -- pinned to the first
    exception of any kind -- so a truncated screencap at photo 1 made
    _is_onnx_provider_failure answer about the wrong exception, no CPU retry happened, and
    the profile was ranked on whatever few photos survived a broken CoreML graph. That is the
    "silently pool a full strength profile embedding out of the ONE surviving photo" failure
    the fallback exists to prevent, reached by photo order instead of by error count."""
    rebuilds, embedder, vec = _run_with_corrupt_photo_at(0)

    assert rebuilds == [["CPUExecutionProvider"]]
    assert embedder._arc_on_cpu is True
    assert vec is not None                       # the 2 decodable photos embedded on CPU
    out = capsys.readouterr().out
    # The corrupt photo still fails on CPU, so the partial-embedding warning stays loud, and
    # the summary reports the whole failed CoreML pass rather than a clean-looking retry.
    assert "WARNING: 1/3 photo(s) failed to embed in this profile" in out
    assert "3 errored on CoreMLExecutionProvider, re-run on CPU" in out


def test_cpu_fallback_is_independent_of_where_the_corrupt_photo_sits():
    """The same profile with the undecodable photo LAST already worked before the fix (the
    first exception was the provider's), so pinning only that case would pass either way.
    Both orderings are asserted together: the fallback decision must be a property of the set
    of errors, not of their arrival order."""
    for position in (0, 1, 2):
        rebuilds, embedder, vec = _run_with_corrupt_photo_at(position)
        assert rebuilds == [["CPUExecutionProvider"]], position
        assert embedder._arc_on_cpu is True, position
        assert vec is not None, position


def test_a_profile_that_only_has_corrupt_photos_never_swaps_the_provider(capsys):
    """The other direction, so the fix cannot be "retry on CPU whenever anything errors": a
    profile whose photos are simply undecodable says nothing about the provider. No rebuild,
    and the pooled embedding is refused outright rather than degraded."""
    embedder = _embedder_on_coreml()
    rebuilds = []

    def fake_build_arc(self, providers):
        rebuilds.append(list(providers))
        return object()

    def fake_embed_image(self, _img):
        raise ValueError(CORRUPT_PHOTO_ERROR)

    embedder._build_arc = types.MethodType(fake_build_arc, embedder)
    embedder._embed_image = types.MethodType(fake_embed_image, embedder)

    vec = embedder.embed_profile(Profile(photos=[b"a", b"b"]))

    assert rebuilds == []
    assert embedder._arc_on_cpu is False
    assert vec is None
    assert "total embedding failure" in capsys.readouterr().out
