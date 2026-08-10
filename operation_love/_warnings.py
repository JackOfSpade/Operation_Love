"""Narrow startup warning filters."""
from __future__ import annotations

import threading
import warnings

_FILTERS = (
    # Still applies: pyproject.toml pins setuptools<82 (see the `ml` extra) precisely so
    # pkg_resources keeps existing for legacy `clip`'s import-time `from pkg_resources import
    # packaging` -- and every setuptools version that still HAS pkg_resources (through 81.0.0)
    # emits this deprecation warning on import. The pin stops the ImportError; this filter just
    # keeps that expected, harmless warning quiet. Remove only if the `setuptools<82` pin (or
    # the clip dependency itself) goes away.
    (r"pkg_resources is deprecated as an API", UserWarning, r"clip(\.|$)"),
    (r"`estimate` is deprecated", FutureWarning, r"insightface\.utils\.face_align$"),
)
_LOCK = threading.Lock()
_CONFIGURED = False


def configure_warnings() -> None:
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED:
            return
        for message, category, module in reversed(_FILTERS):
            warnings.filterwarnings("ignore", message=message, category=category, module=module)
        _CONFIGURED = True
