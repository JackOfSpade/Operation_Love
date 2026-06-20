"""Pick the best available torch device — cross-platform (macOS/Windows/Linux).

Returns "mps" on Apple Silicon, "cuda" on NVIDIA, else "cpu". CPU is fine for
this project's volume, so it runs anywhere even without a GPU. (AMD/ROCm is not
auto-selected — torch falls back to CPU there, which is intentional.)
"""
from __future__ import annotations


def best_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
