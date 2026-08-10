"""Photo quality pre-filter — drop blurry/low-quality shots before scoring.

Uses a pyiqa no-reference metric (default CLIP-IQA) on the best available device
(MPS/CUDA/CPU). The model loads lazily so this module imports without torch/pyiqa
installed; if the library is missing or disabled, it's a no-op (keeps all photos).
The scorer is injectable for tests.
"""
from __future__ import annotations

import threading
from typing import Callable

from ..device import best_device


class QualityFilter:
    # enabled/min_score have no default: both come from config.yaml's quality_filter
    # section (operation_love.config.QualityCfg) at every real call site, so a stale
    # duplicate default here could silently drift from what's actually configured.
    def __init__(self, enabled: bool, min_score: float,
                 metric: str = "clipiqa", scorer: Callable[[bytes], float] | None = None):
        self.enabled = enabled
        self.min_score = min_score
        self.metric = metric
        self._scorer = scorer  # bytes -> score in ~[0,1]
        self._lock = threading.Lock()  # guards double-checked init in _ensure()
        self._init_error: Exception | None = None  # latches a failed init so _ensure() doesn't
                                                     # re-attempt the heavy model load on every photo

    def warmup(self) -> None:
        """Load the IQA model eagerly (call once from the main thread before workers start).

        Raises on failure — any init error (missing deps, SSL, bad model, etc.) is surfaced
        here so the supervisor aborts cleanly rather than silently degrading to no filtering.
        """
        if not self.enabled:
            return
        try:
            self._ensure()
        except Exception as exc:
            print(f"Quality filter FAILED to initialise ({type(exc).__name__}: {exc}); "
                  "aborting run — fix the error or set quality_filter.enabled: false.")
            raise

    def _ensure(self) -> None:
        if self._scorer is not None:               # fast path: no lock needed
            return
        if self._init_error is not None:            # already failed once -> don't retry every photo
            raise self._init_error
        with self._lock:
            if self._scorer is not None:           # second check under lock
                return
            if self._init_error is not None:
                raise self._init_error
            try:
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
            except Exception as exc:  # noqa: BLE001
                self._init_error = exc
                raise

    def score(self, img_bytes: bytes) -> float:
        self._ensure()
        return self._scorer(img_bytes)

    def keep(self, img_bytes: bytes) -> bool:
        if not self.enabled:
            return True
        # Let scoring errors propagate: the worker's halt_on_error machinery will stop the
        # run immediately rather than silently passing every photo through a broken filter.
        return self.score(img_bytes) >= self.min_score

    def filter(self, photos: list[bytes]) -> list[bytes]:
        if not self.enabled:
            return list(photos)
        return [p for p in photos if self.keep(p)]
