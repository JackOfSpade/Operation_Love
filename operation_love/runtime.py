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
from dataclasses import dataclass, field

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
_LEGACY_ALIASES = {"bumble_driver": "android_driver", "hinge_driver": "android_driver"}

# Keep reference/compatibility probes available to diagnostics without turning their absence
# into a startup warning. No registered target currently uses the generic web transport, and
# alias keys would otherwise repeat the same missing ADB dependency two extra times.
_BANNER_HIDDEN_COMPONENTS = frozenset({"web_driver", *_LEGACY_ALIASES})

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
    available: dict[str, bool]
    # Backing field for the `device` property below -- deliberately NOT populated by
    # detect() (see that method's docstring for why). repr=False/compare=False so an
    # unresolved Capabilities still prints and compares the way it always has, with no
    # "_device=None" noise leaking into a banner or a test failure diff.
    _device: str | None = field(default=None, repr=False, compare=False)

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

        Deliberately does NOT resolve `.device` here (see the `device` property) -- every
        _OPTIONAL probe above is a cheap importlib.util.find_spec/shutil.which check, but
        best_device() does a real `import torch` (~600 submodules, ~0.48s measured) to
        decide mps/cuda/cpu, every single time, even for a caller that only wants capability
        flags (a bare diagnostic run, bugreport). That sat badly against this module's
        "quick check on any machine" framing, so the torch import is now deferred to
        whichever caller actually reads `.device` (chiefly `banner()`).
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
            available=available,
        )

    @property
    def device(self) -> str:
        """Best available accelerator (mps/cuda/cpu) -- resolved lazily on first access,
        not by detect() (see detect()'s docstring for why), and cached after that so a
        second read (e.g. a second banner() call) doesn't re-pay the torch import. Callers
        still observe exactly the plain string they always did; only WHEN it's computed
        changed."""
        if self._device is None:
            self._device = best_device()
        return self._device

    def banner(self) -> str:
        accel = _ACCEL.get(self.device, self.device)
        line = f"Operation Love — {self.os_name} {self.machine} · Python {self.python} · {accel}"
        missing = [k for k, v in self.available.items()
                   if not v and k not in _BANNER_HIDDEN_COMPONENTS]
        if missing:
            line += f"\n  Not installed (features will be skipped): {', '.join(missing)}"
        return line

    def missing(self, *components: str) -> list[str]:
        """Of the given components, which are NOT installed."""
        return [c for c in components if not self.available.get(c, False)]


if __name__ == "__main__":
    print(Capabilities.detect().banner())
