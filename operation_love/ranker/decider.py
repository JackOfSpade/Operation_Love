"""Decision layer: quality filter -> embed -> personal ranker.

Turns a captured Profile into an action + the feature embedding (so the swipe is
stored as a training label). Decision values:
  like / dislike  - from the trained PreferenceModel
  no_face         - no detectable face -> can't evaluate (worker passes)
  defer           - model not ready (cold-start): don't swipe blind; seed labels
                    with the labeling tool first, then run autonomously.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from ..perception.capture import Profile


@dataclass
class Decision:
    decision: str                       # like | dislike | no_face | defer
    score: float = 0.0
    embedding: list[float] = field(default_factory=list)  # empty -> no label stored
    source: str = "ranker"


class Decider(Protocol):
    def decide(self, profile: Profile) -> Decision: ...


class RankerDecider:
    """Composes the quality filter, the embedder, and the PreferenceModel."""

    def __init__(self, quality, embedder, model):
        self.quality = quality
        self.embedder = embedder
        self.model = model

    def embed(self, profile: Profile) -> list[float] | None:
        """Quality-filter + embed a profile into a feature vector (no scoring).

        Used in observe mode to turn your manual swipe into a training label.
        None when no face is detected.
        """
        photos = self.quality.filter(profile.photos)
        if not photos:                      # never drop the whole profile on the filter
            photos = profile.photos
        return self.embedder.embed_profile(replace(profile, photos=photos))

    def retrain(self, store) -> bool:
        """Reload all labels and retrain the shared model in place. Returns ready."""
        return self.model.train(store.load_labels())

    def decide(self, profile: Profile) -> Decision:
        vec = self.embed(profile)
        if vec is None:
            return Decision("no_face", 0.0, [], "ranker")
        if not self.model.ready:            # cold-start: collect labels first
            return Decision("defer", 0.0, vec, "cold_start")
        decision, score = self.model.decide(vec)
        return Decision(decision, score, vec, "ranker")
