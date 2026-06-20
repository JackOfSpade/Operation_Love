"""Photo quality pre-filter — drop blurry/low-quality shots before scoring.

Uses a pyiqa no-reference metric (default CLIP-IQA) on the best available device
(MPS/CUDA/CPU). The model loads lazily so this module imports without torch/pyiqa
installed; if the library is missing or disabled, it's a no-op (keeps all photos).
The scorer is injectable for tests.
"""
from __future__ import annotations

from typing import Callable

from ..device import best_device


class QualityFilter:
    def __init__(self, enabled: bool = True, min_score: float = 0.30,
                 metric: str = "clipiqa", scorer: Callable[[bytes], float] | None = None):
        self.enabled = enabled
        self.min_score = min_score
        self.metric = metric
        self._scorer = scorer  # bytes -> score in ~[0,1]

    def _ensure(self) -> None:
        if self._scorer is not None:
            return
        import io

        import pyiqa  # lazy
        import torch
        from PIL import Image
        from torchvision import transforms

        device = best_device()
        model = pyiqa.create_metric(self.metric, device=device)
        to_tensor = transforms.ToTensor()

        def _score(img_bytes: bytes) -> float:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            t = to_tensor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                return float(model(t).item())

        self._scorer = _score

    def score(self, img_bytes: bytes) -> float:
        self._ensure()
        return self._scorer(img_bytes)

    def keep(self, img_bytes: bytes) -> bool:
        if not self.enabled:
            return True
        try:
            return self.score(img_bytes) >= self.min_score
        except Exception:  # noqa: BLE001 - never drop a photo because scoring failed
            return True

    def filter(self, photos: list[bytes]) -> list[bytes]:
        if not self.enabled:
            return list(photos)
        return [p for p in photos if self.keep(p)]
