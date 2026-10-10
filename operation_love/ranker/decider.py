"""Decision layer: quality filter -> embed -> personal ranker.

Turns a captured Profile into an action + the feature embedding (so the swipe is
stored as a training label). Decision values:
  like / dislike  - from the trained PreferenceModel
  no_face         - no detectable face -> can't evaluate (worker passes)
  defer           - model not ready (cold-start): don't swipe blind; seed labels
                    with the labeling tool first, then run autonomously.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol

from ..perception.capture import Profile
from .threshold import fetch_label_rows, fetch_live_review


@dataclass
class Decision:
    decision: str  # like | dislike | no_face | defer
    score: float = 0.0
    embedding: list[float] = field(default_factory=list)  # empty -> no label stored
    source: str = "ranker"


class Decider(Protocol):
    def decide(self, profile: Profile) -> Decision: ...


class RankerDecider:
    """Composes the quality filter, the embedder, and the PreferenceModel."""

    # How long the live auto-review stats may be reused across retrains.  Training mode retrains
    # after EVERY label; the stats only move when the owner answers a review card or AUTO passes
    # someone, and on BigQuery each read is a query.  A minute-old false-pass estimate moves the
    # recall target by a negligible amount (it is a 10+ sample Wilson bound over a 200-answer
    # window), so a short reuse window removes a per-label round trip at no accuracy cost.
    LIVE_REVIEW_TTL_S = 60.0

    def __init__(
        self, quality, embedder, model, *, clock: Callable[[], float] = time.monotonic
    ):
        self.quality = quality
        self.embedder = embedder
        self.model = model
        self._clock = clock
        self._live_review_cache: tuple[int, float, dict] | None = None

    def embed(self, profile: Profile) -> list[float] | None:
        """Quality-filter + embed a profile into a feature vector (no scoring).

        Used in observe mode to turn your manual swipe into a training label.
        None when no face is detected.
        """
        photos = self.quality.filter(profile.photos)
        if not photos:  # never drop the whole profile on the filter
            photos = profile.photos
        return self.embedder.embed_profile(replace(profile, photos=photos))

    def retrain(self, store) -> bool:
        """Reload all labels and retrain the shared model in place. Returns ready.

        Rows carry each label's source (``load_label_rows``) when the store has it, so a
        recall-target model can tell manual calibration labels from answers to auto-pass
        reviews; live review stats feed the model's false-pass correction.  See
        ``threshold.fetch_live_review`` for the best-effort contract of the latter.  A model
        with no recall target never reads them, so the (possibly remote) stats query is skipped.
        """
        rows = fetch_label_rows(store)
        if getattr(self.model, "recall_target", None) is None:
            return self.model.train(rows)
        return self.model.train(rows, live_review=self._live_review(store))

    def _live_review(self, store) -> dict | None:
        """Live review stats, reused for ``LIVE_REVIEW_TTL_S`` (see that constant)."""
        now = self._clock()
        cached = self._live_review_cache
        if (
            cached is not None
            and cached[0] == id(store)
            and 0.0 <= now - cached[1] < self.LIVE_REVIEW_TTL_S
        ):
            return cached[2]
        live = fetch_live_review(store)
        # A failed read (or a store with no review stats) comes back None; never pin that for
        # a minute -- the next retrain asks again.
        self._live_review_cache = (id(store), now, live) if live is not None else None
        return live

    def decide(self, profile: Profile) -> Decision:
        vec = self.embed(profile)
        if vec is None:
            return Decision("no_face", 0.0, [], "ranker")
        if not self.model.ready:  # cold-start: collect labels first
            return Decision("defer", 0.0, vec, "cold_start")
        decision, score = self.model.decide(vec)
        return Decision(decision, score, vec, "ranker")
