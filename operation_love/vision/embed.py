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


class Embedder:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self._arc = None
        self._clip = None
        self._clip_preprocess = None
        self._device = None

    def _ensure(self) -> None:
        if self._arc is not None:
            return
        import open_clip  # lazy
        from insightface.app import FaceAnalysis

        self._device = best_device()
        arc = FaceAnalysis(name="buffalo_l")
        arc.prepare(ctx_id=0 if self._device != "cpu" else -1)
        self._arc = arc

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
            face_vec = [float(x) for x in faces[0].normed_embedding]

        # CLIP (whole image)
        t = self._clip_preprocess(pil).unsqueeze(0).to(self._device)
        with torch.no_grad():
            clip_vec = self._clip.encode_image(t)
            clip_vec = (clip_vec / clip_vec.norm(dim=-1, keepdim=True))[0].cpu().tolist()
        return face_vec, [float(x) for x in clip_vec]

    # --- per-profile ---------------------------------------------------
    def embed_profile(self, profile: Profile) -> list[float] | None:
        self._ensure()
        face_vecs: list[list[float]] = []
        clip_vecs: list[list[float]] = []
        for img in profile.photos:
            try:
                fv, cv = self._embed_image(img)
            except Exception:  # noqa: BLE001
                continue
            if fv is not None:
                face_vecs.append(fv)
            clip_vecs.append(cv)
        if not face_vecs:           # no face anywhere -> can't evaluate the person
            return None
        return concat(aggregate(face_vecs), aggregate(clip_vecs))
