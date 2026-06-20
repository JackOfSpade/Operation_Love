"""Pick the best available torch device.

On the MacBook Pro (Apple M4 Pro) this returns "mps" — PyTorch's Apple-Silicon
GPU backend, which is well supported. Falls back to CUDA (NVIDIA) or CPU.
ROCm/AMD is intentionally not targeted (poor PyTorch support, esp. on Windows).
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
