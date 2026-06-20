"""Decision layer interface.

A Decider turns a captured Profile into a like/dislike with a score and the
feature embedding (so the swipe can be stored as a training label). The real
implementation (local quality filter + ArcFace/CLIP embedding + logistic
regression on your swipes) lands in Phase 3; the protocol lets the worker/
supervisor and tests be built now.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..perception.capture import Profile


@dataclass
class Decision:
    decision: str                       # "like" | "dislike" | "no_face"
    score: float = 0.0                  # P(like) from the ranker
    embedding: list[float] = field(default_factory=list)  # feature vector (empty -> no label stored)
    source: str = "ranker"              # "ranker" | "manual" | "cold_start"


class Decider(Protocol):
    def decide(self, profile: Profile) -> Decision: ...


class RankerDecider:
    """Phase 3: quality pre-filter -> ArcFace+CLIP embedding -> logistic-regression ranker.

    Bootstraps in 'pure personalization' mode: until enough of your own swipe
    labels exist (config.ranker.min_labels_to_engage) it defers to manual
    swiping; after that it scores autonomously.
    """

    def __init__(self, labels: list[tuple[bool, list[float]]], cfg):
        self.labels = labels
        self.cfg = cfg

    def decide(self, profile: Profile) -> Decision:  # pragma: no cover - Phase 3
        raise NotImplementedError("RankerDecider lands in Phase 3 (vision + ranker)")
