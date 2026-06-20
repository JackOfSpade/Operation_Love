"""Pick the best available torch device — cross-platform (macOS/Windows/Linux).

Auto-selects "mps" on Apple Silicon, "cuda" on NVIDIA, else "cpu" (which works
anywhere, GPU or not — fine at this project's volume). Override with the
OPLOVE_DEVICE env var (cpu|cuda|mps) or the ``prefer`` arg. AMD/ROCm is not
auto-selected; torch falls back to CPU there, intentionally.
"""
from __future__ import annotations

import os

_VALID = {"cpu", "cuda", "mps"}


def best_device(prefer: str | None = None) -> str:
    forced = (prefer or os.environ.get("OPLOVE_DEVICE", "")).strip().lower()
    if forced in _VALID:
        return forced

    try:
        import torch
    except Exception:
        return "cpu"

    # Check usability, not just nominal availability.
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
