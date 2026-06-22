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

from ..device import best_device
from ..perception.capture import Profile


def aggregate(vectors: list[list[float]]) -> list[float]:
    """Per-dimension mean across photos. Pure-Python; unit-tested."""
    if not vectors:
        return []
    d = len(vectors[0])
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
    if not vectors:
        return []
    if len(vectors) == 1:
        return list(vectors[0])
    k = len(vectors)
    out: list[float] = []
    for j in range(len(vectors[0])):
        col = [v[j] for v in vectors]
        mag = (sum(abs(x) ** p for x in col) / k) ** (1.0 / p)
        out.append(mag if sum(col) >= 0 else -mag)   # restore the bag's dominant sign
    return out


def dedup_by_cosine(vectors: list[list[float]], threshold: float = 0.85) -> list[list[float]]:
    """Drop near-duplicate vectors (cosine > threshold vs an already-kept one), so a
    burst of near-identical photos can't dominate the pool. Assumes L2-normalized
    inputs (cosine == dot product). Keeps the first of each group. Pure-Python."""
    kept: list[list[float]] = []
    for v in vectors:
        if any(sum(a * b for a, b in zip(v, u, strict=True)) > threshold for u in kept):
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


class Embedder:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self._arc = None
        self._arc_providers: list[str] | None = None
        self._arc_on_cpu = False
        self._clip = None
        self._clip_preprocess = None
        self._device = None

    def _build_arc(self, providers: list[str]):
        from insightface.app import FaceAnalysis

        arc = FaceAnalysis(name="buffalo_l", providers=providers)
        arc.prepare(ctx_id=-1 if providers == ["CPUExecutionProvider"] else 0)
        return arc

    def _set_arc_providers(self, providers: list[str]) -> None:
        self._arc = self._build_arc(providers)
        self._arc_providers = list(providers)
        self._arc_on_cpu = self._arc_providers == ["CPUExecutionProvider"]

    def _ensure(self) -> None:
        if self._arc is not None:
            return
        import open_clip  # lazy
        import onnxruntime as ort

        self._device = best_device()
        providers = _select_onnx_providers(self._device, ort.get_available_providers())
        self._set_arc_providers(providers)

        # Use the -quickgelu variant: the OpenAI weights were trained with QuickGELU,
        # so the plain "ViT-L-14" config (GELU) loads them with a mismatched
        # activation and yields degraded embeddings. Match it for correct CLIP vectors.
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14-quickgelu", pretrained="openai"
        )
        model = model.to(self._device).eval()
        self._clip = model
        self._clip_preprocess = preprocess

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
        faces = self._arc.get(np.array(pil)[:, :, ::-1])  # RGB->BGR
        if faces:
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            # RAW (un-normalized) embedding: its magnitude correlates with face
            # quality, so averaging raws then L2-normalizing is an implicit
            # quality-weighted template (see embed_profile).
            face_vec = [float(x) for x in faces[0].embedding]

        # CLIP (whole image)
        t = self._clip_preprocess(pil).unsqueeze(0).to(self._device)
        with torch.no_grad():
            clip_vec = self._clip.encode_image(t)
            clip_vec = (clip_vec / clip_vec.norm(dim=-1, keepdim=True))[0].cpu().tolist()
        return face_vec, [float(x) for x in clip_vec]

    # --- per-profile ---------------------------------------------------
    def embed_profile(self, profile: Profile) -> list[float] | None:
        self._ensure()
        retried_on_cpu = False
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
                clip_vecs.append(cv)

            provider_failed = (
                bool(profile.photos)
                and errors == len(profile.photos)
                and first_error is not None
                and _is_onnx_provider_failure(first_error)
            )
            if provider_failed and not self._arc_on_cpu and not retried_on_cpu:
                print("CoreML/ONNX provider failed; retrying this profile on CPU "
                      "(CPU stays in effect for the rest of this run).")
                self._set_arc_providers(["CPUExecutionProvider"])
                retried_on_cpu = True
                continue
            break

        # One line per profile so "no_face" is never a silent mystery: how many
        # photos came in, how many had a detectable face, how many errored.
        print(f"Profile: {len(profile.photos)} photo(s) -> "
              f"{len(face_vecs)} with a face"
              f"{f', {errors} errored' if errors else ''}")
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
        return concat(face, clip)
