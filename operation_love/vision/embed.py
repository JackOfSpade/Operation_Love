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

        arc = FaceAnalysis(name="buffalo_l", providers=providers)
        arc.prepare(ctx_id=-1 if providers == ["CPUExecutionProvider"] else 0)
        return arc

    def _set_arc_providers(self, providers: list[str]) -> None:
        self._arc = self._build_arc(providers)
        self._arc_providers = list(providers)
        self._arc_on_cpu = self._arc_providers == ["CPUExecutionProvider"]

    def warmup(self) -> None:
        """Load ML models eagerly (call once from the main thread before workers start)."""
        try:
            self._ensure()
        except Exception as exc:  # noqa: BLE001
            print(f"Embedder warmup failed (will retry per-profile): {type(exc).__name__}: {exc}")

    def _ensure(self) -> None:
        if self._arc is not None:                  # fast path: already init, no lock needed
            return
        with self._lock:
            if self._arc is not None:              # second check under lock (double-checked)
                return
            import open_clip  # lazy
            import onnxruntime as ort

            self._device = best_device()

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
            self._clip = model.to(self._device).eval()
            self._clip_preprocess = preprocess

            providers = _select_onnx_providers(self._device, ort.get_available_providers())
            self._set_arc_providers(providers)     # sets self._arc — LAST, after CLIP succeeded

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

            provider_failed = (
                bool(profile.photos)
                and errors == len(profile.photos)
                and first_error is not None
                and _is_onnx_provider_failure(first_error)
            )
            if provider_failed and not self._arc_on_cpu and not retried_on_cpu:
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

        # One line per profile so "no_face" is never a silent mystery: how many
        # photos came in, how many had a detectable face, how many errored.
        print(f"Profile: {len(profile.photos)} photo(s) -> "
              f"{len(face_vecs)} with a face"
              f"{f', {errors} errored' if errors else ''}")
        # A profile where EVERY photo threw is an operator problem (broken embedder: bad
        # weights, OOM, corrupt install), not a profile property -- left alone it returns
        # None exactly like a genuine no-face profile and silently vanishes into the same
        # "skip" path, with zero signal that the embedder itself is broken. Say so loudly,
        # once per run (same convention as quality.py's _report_failure).
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
