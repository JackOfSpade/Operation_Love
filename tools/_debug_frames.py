"""Containment for a debug-run frame reference named by a release-evidence artifact.

Both release verifiers -- :mod:`tools.hinge_observe_release` (supervised manual) and
:mod:`tools.hinge_observe_ai_release` (AI-driven, explicitly accepted) -- prove parts of their
case by showing that a debug row's ``before``/``after``/``anchor`` names a frame that really was
retained.  Both artifacts unlock the SAME thing: `operation_love/config.py` accepts either as
grounds to start Hinge AUTO.  So the containment rule has to be one rule.  It lived in the
manual verifier only, and the AI one had drifted to a bare ``(run_dir / name).is_file()``, which
a hand-authored ``"../<some-other-run>/anchor.png"`` satisfies -- proving a frame exists
somewhere on the machine rather than that THIS run retained it.

Kept as its own private module rather than imported across the two tools, because the AI
verifier is deliberately a separate evidence chain from the manual one; sharing a leaf
containment predicate must not make either import the other's refusal vocabulary.
"""
from __future__ import annotations

from pathlib import Path


def debug_frame_path(run_dir: Path, name) -> Path | None:
    """Return one retained debug frame only when it is a direct child of ``run_dir``.

    A bare filename, no directory part, no ``.``/``..``, and -- after resolution, so a symlink
    cannot point out of the run -- still a direct child.  Anything else is None, which every
    caller fails closed on.
    """
    if not isinstance(name, str) or not name:
        return None
    relative = Path(name)
    if relative.name != name or name in {".", ".."}:
        return None
    try:
        candidate = run_dir / relative
        if not candidate.is_file() or candidate.resolve().parent != run_dir.resolve():
            return None
    except OSError:
        return None
    return candidate
