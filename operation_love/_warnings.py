"""Narrow startup warning filters."""
from __future__ import annotations

import threading
import warnings

_FILTERS = (
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
