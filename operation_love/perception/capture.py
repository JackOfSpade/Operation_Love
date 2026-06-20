"""The profile data structure passed between layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Profile:
    """Everything captured from one profile.

    photos: raw image bytes (PNG/JPEG) for each photo on the profile.
    prompts: list of (question, answer) for app prompt cards (Hinge/Bumble).
    bio: free-text bio / "about me" if present.
    """
    photos: list[bytes] = field(default_factory=list)
    bio: str = ""
    prompts: list[tuple[str, str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def text_blob(self) -> str:
        parts = [self.bio.strip()] if self.bio else []
        parts += [f"{q.strip()}: {a.strip()}" for q, a in self.prompts if a.strip()]
        return "\n".join(p for p in parts if p)
