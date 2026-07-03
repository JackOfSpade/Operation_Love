"""Local control hub — double-click to open, no terminal needed.

Starts a localhost HTTP server and opens your browser to a control panel that
shows live status for every app and starts/stops runs (observe/auto). Pure
stdlib (http.server), so there's no extra dependency. The run executes in a
background thread in THIS process; the page polls /api/status.

    python -m operation_love hub                  # open the hub
    python -m operation_love hub --make-launchers # write a double-click launcher

The browser is only the face: start/stop POST to this local server, which does
the real work (launch the Bumble browser, and — once an AVD exists — boot the
Hinge emulator). UI choice has no bearing on what the backend can do.

Split into state.py (HubState — the run + browser-liveness bookkeeping),
server.py (the stdlib HTTP handler + process lifecycle), launchers.py
(OS-specific double-click launcher scripts), and page.py (the inlined
frontend). This module re-exports the full former flat-module surface so
existing imports (`from .hub import make_launchers, serve`) and tests
(`from operation_love.hub import HubState, _Handler, ...`) keep working
unchanged.
"""
from __future__ import annotations

import time

from .. import config as cfg_mod
from .. import supervisor

from .launchers import _LINUX_UPDATE_RUN, _MAC_UPDATE_RUN, _WIN_UPDATE_RUN, make_launchers
from .page import _PAGE
from .server import _BROWSER_SHUTDOWN_GRACE_S, _BROWSER_STALE_CHECK_S, _Handler, _bind, serve
from .state import _BROWSER_CLIENT_STALE_S, _CLOSED_BROWSER_CLIENT_TTL_S, HubState

__all__ = [
    "HubState", "_Handler", "_bind", "serve", "make_launchers",
    "_MAC_UPDATE_RUN", "_LINUX_UPDATE_RUN", "_WIN_UPDATE_RUN", "_PAGE",
    "_BROWSER_SHUTDOWN_GRACE_S", "_BROWSER_CLIENT_STALE_S",
    "_BROWSER_STALE_CHECK_S", "_CLOSED_BROWSER_CLIENT_TTL_S",
    "cfg_mod", "supervisor", "time",
]
