"""Feature embeddings: ArcFace (facial structure) + CLIP (style/vibe).

For a profile we embed each photo and aggregate into one feature vector that the
PreferenceModel scores. ArcFace captures facial structure; CLIP captures overall
look/style the face vector misses — concatenating both is the design the research
converged on. Heavy libs (insightface, open_clip, torch) load lazily on the best
device, so this module imports anywhere; real embedding runs on a machine with
the `ml` extra installed.

Returns None for a profile with no detectable face (-> "no_face" decision).
"""
from __future__ import annotations

import math
import threading

from ..device import best_device
from ..perception.capture import Profile


def _finite_builtin_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _vector_dimension(vectors: list[list[float]]) -> int:
    try:
        dimension = len(vectors[0])
    except (IndexError, TypeError) as exc:
        raise ValueError("vectors must contain nonempty sequences") from exc
    if dimension == 0:
        raise ValueError("vectors must have a nonzero dimension")
    for ordinal, vector in enumerate(vectors):
        try:
            actual = len(vector)
        except TypeError as exc:
            raise ValueError(f"vector {ordinal} is not a sequence") from exc
        if actual != dimension:
            raise ValueError(
                f"vector {ordinal} has dimension {actual}; expected {dimension}")
    return dimension


def aggregate(vectors: list[list[float]]) -> list[float]:
    """Per-dimension mean across photos. Pure-Python; unit-tested."""
    if not vectors:
        return []
    d = _vector_dimension(vectors)
    n = len(vectors)
    return [sum(v[j] for v in vectors) / n for j in range(d)]


def concat(*parts: list[float]) -> list[float]:
    out: list[float] = []
    for p in parts:
        out.extend(p)
    return out


def l2_normalize(vec: list[float]) -> list[float]:
    """Scale to unit length (no-op for a ~zero vector). Pooling changes a vector's
    norm; re-normalizing puts it back on the unit hypersphere so the linear model
    sees the pretrained cosine geometry. Pure-Python; unit-tested."""
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n > 1e-12 else list(vec)


def gem_pool(vectors: list[list[float]], p: float = 3.0) -> list[float]:
    """Generalized-mean (GeM) pool across a set of vectors, per dimension.

    Amplifies dimensions strongly active in ANY photo (salient style cues) without
    the full noise-sensitivity of max pooling: p=1 is the mean, p->inf approaches
    max. Sign-preserving so it works on CLIP's signed features. Pure-Python.
    """
    exponent = _finite_builtin_number(p)
    if exponent is None or exponent <= 0:
        raise ValueError("p must be a positive finite number")
    if not vectors:
        return []
    dimension = _vector_dimension(vectors)
    if len(vectors) == 1:
        return list(vectors[0])
    k = len(vectors)
    out: list[float] = []
    for j in range(dimension):
        col = [v[j] for v in vectors]
        mag = (sum(abs(x) ** exponent for x in col) / k) ** (1.0 / exponent)
        out.append(mag if sum(col) >= 0 else -mag)   # restore the bag's dominant sign
    return out


def square_crop_around_bbox(size: tuple[int, int], bbox, expand: float = 3.5) -> tuple[int, int, int, int]:
    """A square crop box (left, top, right, bottom) centered on a face bbox and expanded to
    include the upper body / setting, clamped inside the image. Used to give CLIP a person-centric
    ~square photo crop instead of the full screenshot (status bar, buttons, white margins), which
    otherwise dilutes the style/vibe signal. Pure (takes image size + bbox); unit-tested."""
    w, h = size
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # at least 2px (never a degenerate empty crop), at most the image's min dimension
    side = min(max(max(x2 - x1, y2 - y1) * expand, 2.0), float(min(w, h)))
    half = side / 2.0
    cx = min(max(cx, half), w - half)                              # keep the square inside the image
    cy = min(max(cy, half), h - half)
    left, top = int(round(cx - half)), int(round(cy - half))
    return (left, top, int(round(left + side)), int(round(top + side)))


def dedup_by_cosine(vectors: list[list[float]], threshold: float = 0.85) -> list[list[float]]:
    """Drop near-duplicate vectors (cosine > threshold vs an already-kept one), so a
    burst of near-identical photos can't dominate the pool. Assumes L2-normalized
    inputs (cosine == dot product). Keeps the first of each group. Pure-Python."""
    threshold_value = _finite_builtin_number(threshold)
    if threshold_value is None or not -1.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be a finite number in [-1, 1]")
    if not vectors:
        return []
    _vector_dimension(vectors)
    kept: list[list[float]] = []
    for v in vectors:
        if any(
                sum(a * b for a, b in zip(v, u, strict=True)) > threshold_value
                for u in kept):
            continue
        kept.append(v)
    return kept


def _select_onnx_providers(device: str, available) -> list[str]:
    available_set = set(available or ())
    if device in {"mps", "apple", "coreml"}:
        preferred = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    elif device == "cuda":
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        preferred = ["CPUExecutionProvider"]
    selected = [provider for provider in preferred if provider in available_set]
    # Never hand onnxruntime an empty provider list (it raises); CPU is always valid.
    return selected or ["CPUExecutionProvider"]


# CoreML EP options, passed for CoreMLExecutionProvider only (see _onnx_provider_options).
# ModelFormat=MLProgram instead of onnxruntime's default NeuralNetwork, measured 2026-08-11 on
# this project's dev Mac (Apple Silicon, macOS 26, onnxruntime 1.27.0, insightface 1.0.1) on
# buffalo_l's recognition model w600k_r50, one 112x112 crop per call:
#     CPU                          50.8 ms   (baseline)
#     CoreML, MLProgram             5.5 ms   cosine vs CPU 0.99999994 (worst of 5 inputs)
#     CoreML, NeuralNetwork         2.0 ms   cosine vs CPU 0.9957     (worst of 5 inputs)
# NeuralNetwork is the faster of the two because it runs fp16 on the ANE, and that is exactly
# why it is rejected: the face embedding is the ranker's input, and every stored label in the
# training set was collected against CPU-computed vectors. Shifting the feature space under a
# trained model to save 3.5ms/photo is a silent degradation, which the owner rule forbids.
# MLProgram is ~9x faster than CPU at fp32 parity, so take that.
_COREML_PROVIDER_OPTIONS = {"ModelFormat": "MLProgram"}


def _onnx_provider_options(providers) -> list[dict[str, str]]:
    """Per-provider option dicts, positionally aligned with `providers`.

    onnxruntime (and insightface, which forwards `provider_options` straight through to
    ort.InferenceSession) requires this list to be the same length as the provider list and
    matched by position, so every non-CoreML provider gets an empty dict rather than being
    omitted. Pure; unit-tested.
    """
    return [
        dict(_COREML_PROVIDER_OPTIONS) if p == "CoreMLExecutionProvider" else {}
        for p in providers
    ]


def _pin_detector_to_cpu(arc) -> None:
    """Move ONLY buffalo_l's face DETECTOR (det_10g) off CoreML; leave every other model on it.

    This is the fix for the error that used to abort the CoreML attempt on every single run:

        [ONNXRuntimeError] : 1 : FAIL : ... CoreMLExecutionProvider ... GetStaticOutputShape
        ... CoreML static output shape ({1,1,1,128,1}) and inferred shape ({3200,1}) have
        different ranks.

    Diagnosed offline 2026-08-11 (onnxruntime 1.27.0, insightface 1.0.1, Apple Silicon), and it
    is a configuration problem, not an unfixable model/provider incompatibility:

    det_10g.onnx takes a DYNAMIC input ([1, 3, '?', '?']) but DECLARES its output shapes for a
    640x640 input -- 12800/3200/800 rows, one per FPN stride. Feed it any other size and the
    real output disagrees with the declared one. The CPU EP shrugs (a VerifyOutputSizes warning,
    suppressed anyway because insightface sets the ort log severity to ERROR) and returns the
    true shape; the CoreML EP hard-asserts on the mismatch instead. The {1,1,1,128,1} in the
    message is the rank-5 CoreML form of the 128 rows a 128x128 input really produces, against
    the 3200 rows the graph declares. Probed size by size on CoreML: 640 OK; 128, 320 and 1024
    all FAIL with that assert.

    That is the trap insightface 1.0.1 walks into. FaceAnalysis.prepare() with no det_size now
    defaults to MULTI-SCALE detection (DEFAULT_DET_SIZES = [(128,128), (640,640)]) and
    SCRFD.detect() runs the session at BOTH sizes on every call -- so the 128x128 pass fails on
    CoreML on the very first inference, always, regardless of the photo.

    Two ways out, and the choice matters. Pinning det_size=(640,640) would let the detector run
    on CoreML (53.8ms -> 17.3ms per call), but it silently drops the 128x128 scale and therefore
    changes which faces are detected at all -- buying speed with detection recall, on the input
    to a trained ranker, for a "no face -> skip this profile" decision. Not acceptable under the
    never-silently-degrade rule. So do the opposite: leave the detector's input sizes exactly as
    insightface chose them and give it the provider that tolerates them. Detection behaviour is
    then bit-for-bit identical to what ships today; only its provider is now stated deliberately
    instead of being discovered by a failed attempt and a fallback on every run.

    (Also measured and rejected: provider option RequireStaticInputShapes=1, which keeps CoreML
    off dynamically-shaped nodes and does make the detector safe at both scales -- but
    w600k_r50's batch dim is symbolic too, so it pushes recognition back to 48.5ms/call, i.e. it
    buys nothing over plain CPU. The other three buffalo_l models -- 2d106det, 1k3d68,
    genderage -- were each checked on CoreML/MLProgram and agree with CPU to cosine 1.0000000,
    so the detector is the only one that needs this.)

    set_providers() on the live session is insightface's own mechanism for exactly this
    (SCRFD.prepare does the same thing when ctx_id < 0), and FaceAnalysis.__init__ asserts
    'detection' is present, so the lookup below cannot silently no-op. Call this BEFORE
    arc.prepare(): prepare() with ctx_id >= 0 does not touch providers, so the pin survives it.
    """
    arc.models["detection"].session.set_providers(["CPUExecutionProvider"])


def _is_onnx_provider_failure(error: BaseException | str) -> bool:
    msg = str(error)
    upper = msg.upper()
    return (
        "COREML" in upper
        or "ONNXRUNTIMEERROR" in upper
        or "STATUS FAIL" in upper
        or "STATUS: FAIL" in upper
        or " : FAIL :" in upper
    )


def _synthetic_probe_image() -> bytes:
    """A tiny (128x128) PNG generated entirely in memory -- never read from disk or the
    network -- used only to warm up a REAL inference call (see Embedder._probe_inference).
    Fixed seed: the probe's job is to exercise the provider, not to test face detection,
    so deterministic content (identical across every run/machine) is preferable to random
    content that would make a probe failure harder to reproduce. Structured noise rather
    than a flat fill, so the image isn't degenerate enough for some detector graph to
    short-circuit before reaching the same conv/reshape ops a real profile photo would.
    Pure (numpy+PIL, no I/O); unit-tested.
    """
    import io

    import numpy as np
    from PIL import Image

    arr = np.random.default_rng(0).integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class Embedder:
    def __init__(self):
        self._arc = None
        self._arc_providers: list[str] | None = None
        self._arc_on_cpu = False
        self._clip = None
        self._clip_preprocess = None
        self._device = None
        self._lock = threading.Lock()  # guards double-checked init in _ensure()
        self._reported_total_failure = False  # surface a total-failure profile once, not per profile

    def _build_arc(self, providers: list[str]):
        from insightface.app import FaceAnalysis

        arc = FaceAnalysis(name="buffalo_l", providers=providers,
                           provider_options=_onnx_provider_options(providers))
        if "CoreMLExecutionProvider" in providers:
            # Before prepare(), and only when CoreML is actually in play: the detector cannot
            # run there at insightface's multi-scale det sizes. See _pin_detector_to_cpu.
            _pin_detector_to_cpu(arc)
        arc.prepare(ctx_id=-1 if providers == ["CPUExecutionProvider"] else 0)
        return arc

    def _set_arc_providers(self, providers: list[str]) -> None:
        self._arc = self._build_arc(providers)
        self._arc_providers = list(providers)
        self._arc_on_cpu = self._arc_providers == ["CPUExecutionProvider"]

    def warmup(self) -> None:
        """Load ML models eagerly (call once from the main thread before workers start).

        Raises on failure -- matches QualityFilter.warmup()'s contract (vision/quality.py),
        which this project's fail-loud rule requires: never silently degrade. This used to
        catch-and-print any init error (missing weights, a network hiccup downloading them,
        OOM, a corrupt onnxruntime install, ...) and let the run continue, on the theory
        that embed_profile()'s own _ensure() call would "retry per-profile" -- but that
        retry runs unguarded, inside a worker thread, only after supervisor.run() has
        already finished the rest of its (possibly slow) startup, taken the Android device
        lock, and opened a live session. So a broken embedder wasn't actually retried
        gracefully: it sailed straight through, captured a REAL profile off the phone, and
        only then failed on that profile's embed. Letting the exception propagate here
        means supervisor.run() aborts cleanly during startup instead, before any of that.
        (A missing ML extra specifically -- arcface/clip literally not installed -- is
        caught earlier and separately, by supervisor.run()'s own
        `caps.missing("arcface", "clip")` gate; this covers every OTHER init failure that
        gate can't see, since the libraries can be installed and still fail to load.)

        _ensure() above only BUILDS the ONNX session (ort.InferenceSession(...) /
        insightface's arc.prepare(...)) -- neither call actually runs the model, so a
        provider that builds cleanly can still be unable to INFER. Observed on this
        project's dev Mac (Apple Silicon): CoreMLExecutionProvider built buffalo_l's
        session with no error, but the compiled graph had a static-shape incompatibility
        that only threw on the first real `.get()` call, well after warmup had already
        reported success -- see _probe_inference()'s docstring for the exact error and
        why running one real synthetic inference here, through the SAME fallback
        embed_profile() already has, is what catches it during startup instead of on the
        operator's first real profile. (That specific incompatibility is now fixed at its
        source -- _pin_detector_to_cpu() -- so the probe should no longer trip on it; the
        probe stays because "builds fine, cannot infer" is a provider-class hazard, not a
        one-off bug, and it is the only thing standing between a future recurrence and the
        operator's first real profile.)
        """
        self._ensure()
        self._probe_inference()

    def _ensure(self) -> None:
        if self._arc is not None:                  # fast path: already init, no lock needed
            return
        with self._lock:
            if self._arc is not None:              # second check under lock (double-checked)
                return
            import open_clip  # lazy
            import onnxruntime as ort

            device = best_device()

            # Build CLIP FIRST and commit self._arc LAST. self._arc is the sentinel the guard
            # above short-circuits on, so it must only be set once BOTH models have loaded. If
            # CLIP's load raised (weight download / OOM) AFTER _arc were set, the next _ensure()
            # would short-circuit on a half-initialized embedder (_clip still None) and every
            # _embed_image would TypeError -> silently "no_face" for the rest of the run.
            # Use the -quickgelu variant: the OpenAI weights were trained with QuickGELU,
            # so the plain "ViT-L-14" config (GELU) loads them with a mismatched
            # activation and yields degraded embeddings. Match it for correct CLIP vectors.
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14-quickgelu", pretrained="openai"
            )
            clip = model.to(device).eval()

            providers = _select_onnx_providers(device, ort.get_available_providers())
            arc = self._build_arc(providers)
            # Publish the complete model bundle atomically under the lock.  In particular, an
            # ArcFace build failure must not retain a heavyweight CLIP object or a device from a
            # half-finished attempt; the next _ensure() starts from the same clean state.
            self._device = device
            self._clip = clip
            self._clip_preprocess = preprocess
            self._arc_providers = list(providers)
            self._arc_on_cpu = self._arc_providers == ["CPUExecutionProvider"]
            self._arc = arc                    # initialization sentinel is committed last
            # Say what ArcFace actually ended up on, once, at init. Before the detector pin
            # the operator's only evidence was a CoreML stack trace at warmup followed by a
            # silently CPU-only session; a provider split this consequential (~9x on the
            # recognition model) should be legible from the run log, not only from the code.
            detector_note = (" (detector on CPUExecutionProvider — see _pin_detector_to_cpu)"
                             if "CoreMLExecutionProvider" in providers else "")
            print(f"ArcFace providers: {', '.join(providers)}{detector_note}")

    def _probe_inference(self) -> None:
        """Run ONE real inference through embed_profile()'s exact code path, on a tiny
        synthetic image, so a provider that BUILT successfully but cannot actually INFER
        is caught here during startup instead of on the operator's first real profile.

        Concrete motivating case (a real observe-mode run on this project's dev Mac,
        Apple Silicon, verified against this code), since fixed at its source by
        _pin_detector_to_cpu() but kept here because it is what this probe is shaped
        around: CoreMLExecutionProvider's ort.InferenceSession(...) for buffalo_l builds
        without error -- the eager _ensure() call warmup() makes above sees a clean
        success -- but the compiled graph has a static-shape incompatibility that only
        throws on the first real `.get()` call:
            Photo embedding error: Fail: [ONNXRuntimeError] : 1 : FAIL : ...
            CoreMLExecutionProvider ... Status Message: Exception: ...
            GetStaticOutputShape ... CoreML static output shape ({1,1,1,128,1}) and
            inferred shape ({3200,1}) have different ranks.
        Previously that was invisible until the operator's first real profile: it burned
        ~55s embedding all 8 of that profile's photos, errored on every one, and only
        THEN rebuilt the whole FaceAnalysis stack and re-embedded on CPU -- mid-run,
        after supervisor.run() had already taken the Android device lock and opened a
        live session (see warmup()'s docstring above for why that ordering matters).

        Routed through embed_profile() itself, not a parallel check: a provider-shaped
        failure here hits the SAME CPU-fallback branch, under the SAME lock, with the
        SAME loud prints, that a real profile's failure would -- one fallback path to
        keep correct, not two. embed_profile() never raises on a photo's embedding
        error (it counts it, prints it, and continues), so a non-provider-shaped hiccup
        on this synthetic image cannot turn into a hard startup failure here either --
        only a genuine _ensure() failure above (unchanged) does that.

        Known limit, stated rather than papered over: structured noise contains no face,
        so this probe exercises buffalo_l's DETECTOR only. The recognition and landmark
        models run per detected face, so nothing here can reach them -- and after
        _pin_detector_to_cpu() they are precisely the models left on CoreML. Their safety
        net is therefore still embed_profile()'s runtime fallback: one profile re-embedded
        on CPU, loudly, never a wrong decision. Making the probe reach them would mean
        shipping an image a face detector is guaranteed to fire on, which is not something
        a synthetic fixture can promise across model versions.
        """
        self.embed_profile(Profile(photos=[_synthetic_probe_image()]))

    # --- per-photo -----------------------------------------------------
    def _embed_image(self, img_bytes: bytes) -> tuple[list[float] | None, list[float]]:
        """Return (arcface_vec or None if no face, clip_vec)."""
        import io

        import numpy as np
        import torch
        from PIL import Image

        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ArcFace (largest detected face)
        face_vec = None
        clip_src = pil                                    # default: whole image (no face -> skipped anyway)
        faces = self._arc.get(np.array(pil)[:, :, ::-1])  # RGB->BGR
        if faces:
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            # RAW (un-normalized) embedding: its magnitude correlates with face
            # quality, so averaging raws then L2-normalizing is an implicit
            # quality-weighted template (see embed_profile).
            face_vec = [float(x) for x in faces[0].embedding]
            # CLIP sees a person-centric SQUARE crop around the face, not the whole screenshot:
            # ArcFace already self-crops the face; this keeps CLIP's style/vibe signal on the
            # actual photo instead of UI chrome + margins (the screenshot is host-side, no
            # element selectors, so the photo is a fraction of the frame).
            try:
                clip_src = pil.crop(square_crop_around_bbox(pil.size, faces[0].bbox))
            except Exception:  # noqa: BLE001 — bad bbox -> fall back to the whole image
                clip_src = pil

        # CLIP (person-centric square crop when a face was found, else the whole image)
        t = self._clip_preprocess(clip_src).unsqueeze(0).to(self._device)
        with torch.no_grad():
            clip_vec = self._clip.encode_image(t)
            clip_vec = (clip_vec / clip_vec.norm(dim=-1, keepdim=True))[0].cpu().tolist()
        return face_vec, [float(x) for x in clip_vec]

    # --- per-profile ---------------------------------------------------
    def embed_profile(self, profile: Profile) -> list[float] | None:
        self._ensure()
        retried_on_cpu = False
        prior_errors = 0             # errors from a failed pass BEFORE a CPU retry, if any
        prior_provider: str | None = None
        while True:
            face_vecs: list[list[float]] = []
            clip_vecs: list[list[float]] = []
            first_error: Exception | None = None
            errors = 0
            for img in profile.photos:
                try:
                    fv, cv = self._embed_image(img)
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    if first_error is None:
                        first_error = exc
                        # Surface the first failure; don't spam one line per photo.
                        print(f"Photo embedding error: {type(exc).__name__}: {exc}")
                    continue
                if fv is not None:
                    face_vecs.append(fv)
                    # Only pool CLIP when a face was found: with no face, _embed_image's
                    # clip_src falls back to the WHOLE screenshot (status bar, buttons, white
                    # margins), not the person-centric crop square_crop_around_bbox exists to
                    # produce. Mixing that in would dilute the pooled style/vibe vector with UI
                    # chrome. A faceless photo just contributes nothing here (a profile with NO
                    # faces at all still returns None below, so no photo signal is silently lost).
                    clip_vecs.append(cv)

            # A provider-shaped error on ANY photo (not just every photo) means the
            # embedder itself is broken for this provider, not that a few photos happen
            # to be bad: CoreMLExecutionProvider's static-shape graph on this machine
            # (see _probe_inference's docstring for the exact rank-mismatch error) throws
            # the SAME exception on every call it's given, so errors < len(photos) here
            # just means the run got lucky about which photos ran before the first
            # failure, not that the provider is partly working. Gating the fallback on
            # errors == len(photos) let a 7-of-8 CoreML failure stay on the broken
            # provider and silently pool a "full strength" profile embedding out of the
            # ONE surviving photo -- fail loud instead: retry the WHOLE profile on CPU.
            provider_failed = (
                bool(profile.photos)
                and errors > 0
                and first_error is not None
                and _is_onnx_provider_failure(first_error)
            )
            if provider_failed and not self._arc_on_cpu and not retried_on_cpu:
                # Capture what this pass actually ran on and how many photos it lost,
                # BEFORE the swap below overwrites _arc_providers, so the summary line
                # after the loop can report the failed first pass even after a
                # successful CPU retry (see the retry_note below).
                prior_errors = errors
                prior_provider = (self._arc_providers or ["unknown provider"])[0]
                # Guard the provider swap with the same lock _ensure() uses: two Workers can
                # hit this concurrently, and without the lock both would observe
                # not self._arc_on_cpu and both rebuild the full FaceAnalysis model (duplicated
                # multi-second loads + a torn read of _arc_on_cpu). Re-check under the lock so
                # only the winner rebuilds; the loser just retries its own profile against the
                # now-CPU model the winner installed. Scope stays tight — no inference in here.
                with self._lock:
                    if not self._arc_on_cpu:
                        print("CoreML/ONNX provider failed; retrying this profile on CPU "
                              "(CPU stays in effect for the rest of this run).")
                        self._set_arc_providers(["CPUExecutionProvider"])
                retried_on_cpu = True
                continue
            break

        # One line per profile so "no_face" is never a silent mystery: how many photos
        # came in, how many had a detectable face, how many errored on THIS (possibly
        # CPU-retried) pass -- plus, if an earlier pass on a different provider failed
        # before the retry kicked in, how many that pass lost too. Without carrying
        # prior_errors/prior_provider across the retry, a profile whose first pass
        # errored 8/8 on CoreML and then succeeded 8/8 on CPU read as a perfectly clean
        # "8 photo(s) -> 8 with a face", with zero record that the first pass had failed
        # outright.
        retry_note = (
            f" ({prior_errors} errored on {prior_provider}, re-run on CPU)" if prior_errors else ""
        )
        print(f"Profile: {len(profile.photos)} photo(s) -> "
              f"{len(face_vecs)} with a face"
              f"{f', {errors} errored' if errors else ''}"
              f"{retry_note}")
        # Any photo that fails to embed and STAYS failed (no further fallback recovered
        # it) means the pooled embedding below is built from fewer photos than the
        # profile actually has -- a partially-embedded profile must never look
        # indistinguishable from a clean one in the run's output, whether it's every
        # photo (a fully broken embedder) or just some of them (a provider already on
        # CPU with nowhere left to fall back to, or a handful of corrupt/undecodable
        # images unrelated to the provider).
        if errors:
            print(f"WARNING: {errors}/{len(profile.photos)} photo(s) failed to embed in "
                  f"this profile ({type(first_error).__name__}: {first_error}) -- the "
                  "pooled embedding below is built from fewer photos than the profile "
                  "actually has.")
        # A profile where EVERY photo threw (on the final pass) is an operator problem
        # (broken embedder: bad weights, OOM, corrupt install), not a profile property --
        # left alone it returns None exactly like a genuine no-face profile and silently
        # vanishes into the same "skip" path, with zero signal that the embedder itself
        # is broken. Say so loudly, once per run (same convention as quality.py's
        # _report_failure).
        if bool(profile.photos) and errors == len(profile.photos) and not self._reported_total_failure:
            self._reported_total_failure = True
            print(f"WARNING: total embedding failure ({errors}/{len(profile.photos)} photos "
                  f"errored: {type(first_error).__name__}: {first_error}) -- this looks like a "
                  "broken embedder, not a genuine no-face profile. (further occurrences this "
                  "run are not logged)")
        if not face_vecs:           # no face anywhere -> can't evaluate the person
            return None
        # Modality-specific aggregation (per the small-data MIL research). All of a
        # profile's photos collapse into ONE vector, so the label is per PROFILE.
        #  - ArcFace identity is ~constant across a profile's photos: the mean of the
        #    RAW face vectors, L2-normalized, is a denoised, implicitly
        #    quality-weighted template (better faces carry larger raw norms).
        #  - CLIP style/vibe varies across photos: drop near-duplicate bursts, then
        #    GeM-pool (p=3) to keep salient cues instead of averaging them away.
        # Each modality is L2-normalized BEFORE concatenation so the linear model
        # weights them equally and the pretrained cosine geometry is preserved.
        face = l2_normalize(aggregate(face_vecs))
        clip = l2_normalize(gem_pool(dedup_by_cosine(clip_vecs)))
        vec = concat(face, clip)
        if not all(math.isfinite(x) for x in vec):
            print("Profile embedding contains non-finite values (NaN/Inf); treating as no_face.")
            return None
        return vec
