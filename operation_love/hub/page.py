"""The hub's single-page HTML/CSS/JS control panel.

Kept separate from the request-handling and state/process-lifecycle code so
the frontend can be edited without touching either. The page itself lives in
../assets/hub.html as a real file, not inlined here as a Python string: that
gets real editor tooling (syntax highlighting, linting, formatting) and,
critically, no Python-level string escaping — a doubled backslash in a Python
string is an easy, silent way to corrupt the JS. Do not inline it back.
"""
from __future__ import annotations

from pathlib import Path

_PAGE = (Path(__file__).parent.parent / "assets" / "hub.html").read_text(encoding="utf-8")
