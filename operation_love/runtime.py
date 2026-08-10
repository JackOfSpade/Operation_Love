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
#
# Keyed by TRANSPORT (how we drive a platform — see operation_love/platforms.py), not by
# dating app: since Aug 2026 both Hinge and Bumble drive the same physical Pixel over
# host-side ADB, so one "android_driver" check covers either app rather than a per-app key
# that would falsely suggest Bumble needs something different from Hinge. "web_driver"
# similarly covers whichever web-based platform (if any) is live, not just Bumble.
_OPTIONAL = {
    "torch": "torch",
    "clip": "open_clip",
    "arcface": "insightface",
    "quality": "pyiqa",
    "web_driver": "playwright",
    "android_driver": "cli:adb",      # host-side ADB only — no uiautomator2/on-device helper
    # Vision. Listed so a preflight/bug report can SAY "opencv is missing" rather than the
    # operator finding out when a run refuses to start. The Android driver independently
    # hard-fails open_session() without it (AndroidDriver._require_vision) — this entry is
    # for visibility, not safety. It earned its place: a launcher once shipped without the
    # `hinge` extra, so cv2 was absent and every template match quietly returned nothing.
    "vision_templates": "cv2",
    "bigquery": "google.cloud.bigquery",
    "cloud_storage": "google.cloud.storage",
}

# Pre-rename aliases, kept so any caller still reading the old per-app keys degrades
# gracefully instead of KeyError-ing. Populated onto `available` in detect() below.
_LEGACY_ALIASES = {"bumble_driver": "web_driver", "hinge_driver": "android_driver"}

_ACCEL = {"mps": "Apple GPU (MPS)", "cuda": "NVIDIA GPU (CUDA)", "cpu": "CPU"}


def _have(mod: str, *, override: str | None = None) -> bool:
    if mod.startswith("cli:"):
        # override lets a caller point a "cli:" check at a configured binary path instead
        # of the bare command name (e.g. apps.hinge.adb_path when adb isn't on PATH).
        # shutil.which handles both: a bare name searches PATH, a path with a separator
        # is checked directly.
        return shutil.which(override or mod[4:]) is not None
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
    def detect(cls, android_adb_path: str | None = None, *,
               hinge_adb_path: str | None = None) -> "Capabilities":
        """android_adb_path: optional apps.<app>.adb_path from config.yaml for whichever
        enabled app is Android-kind (Hinge, or Bumble once calibrated — see platforms.py).
        Config-free callers (bare `python -m operation_love.runtime`, bugreport) can omit
        it and get today's PATH-only check; a caller that has loaded config should pass it
        so a machine with adb configured off-PATH doesn't get a false "not installed".

        hinge_adb_path is the pre-rename kwarg, kept as a backwards-compatible alias for
        android_adb_path (Hinge was the only Android app when it was named); prefer the
        new name in new code.
        """
        android_adb_path = android_adb_path or hinge_adb_path
        overrides = {"android_driver": android_adb_path} if android_adb_path else {}
        available = {k: _have(m, override=overrides.get(k)) for k, m in _OPTIONAL.items()}
        for old, new in _LEGACY_ALIASES.items():
            available[old] = available[new]
        return cls(
            os_name=platform.system() or "unknown",
            machine=platform.machine() or "unknown",
            python=platform.python_version(),
            device=best_device(),
            available=available,
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
