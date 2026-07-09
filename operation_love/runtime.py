"""Detect the runtime environment so the app auto-adjusts to any OS / hardware.

Call ``Capabilities.detect()`` once at startup: ``.device`` tells the ML layers
which accelerator to use, and ``.available`` says which optional components are
installed so the supervisor can degrade gracefully (skip the quality filter,
disable openers, etc.) instead of crashing on a machine missing a dependency.

Quick check on any machine:  ``python -m operation_love.runtime``
"""
from __future__ import annotations

import importlib.util
import platform
import shutil
from dataclasses import dataclass

from .device import best_device

# component key -> import name it needs (or "cli:<binary>" for a required CLI tool)
_OPTIONAL = {
    "torch": "torch",
    "clip": "open_clip",
    "arcface": "insightface",
    "quality": "pyiqa",
    "bumble_driver": "playwright",
    "hinge_driver": "cli:adb",        # host-side ADB only — no uiautomator2/on-device helper
    "anthropic": "anthropic",
    "bigquery": "google.cloud.bigquery",
    "cloud_storage": "google.cloud.storage",
}

_ACCEL = {"mps": "Apple GPU (MPS)", "cuda": "NVIDIA GPU (CUDA)", "cpu": "CPU"}


def _have(mod: str) -> bool:
    if mod.startswith("cli:"):
        return shutil.which(mod[4:]) is not None
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


@dataclass
class Capabilities:
    os_name: str
    machine: str
    python: str
    device: str
    available: dict[str, bool]

    @classmethod
    def detect(cls) -> "Capabilities":
        return cls(
            os_name=platform.system() or "unknown",
            machine=platform.machine() or "unknown",
            python=platform.python_version(),
            device=best_device(),
            available={k: _have(m) for k, m in _OPTIONAL.items()},
        )

    def banner(self) -> str:
        accel = _ACCEL.get(self.device, self.device)
        line = f"Operation Love — {self.os_name} {self.machine} · Python {self.python} · {accel}"
        missing = [k for k, v in self.available.items() if not v]
        if missing:
            line += f"\n  Not installed (features will be skipped): {', '.join(missing)}"
        return line

    def missing(self, *components: str) -> list[str]:
        """Of the given components, which are NOT installed."""
        return [c for c in components if not self.available.get(c, False)]


if __name__ == "__main__":
    print(Capabilities.detect().banner())
