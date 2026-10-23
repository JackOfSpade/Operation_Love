"""Decision layer: quality filter -> embed -> personal ranker.

Turns a captured Profile into an action + the feature embedding (so the swipe is
stored as a training label). Decision values:
  like / dislike  - from the trained PreferenceModel
  no_face         - no detectable face -> can't evaluate (worker passes)
  defer           - model not ready (cold-start): don't swipe blind; seed labels
                    with the labeling tool first, then run autonomously.

Feature sets (``ranker.feature_set``):
  face_crop_v1              - the 1280-d face-crop vector alone (the default).
  face_crop_whole_photo_v1  - that vector followed by the 768-d whole-photo vector, which pools
                              EVERY unique photo card of the profile (the owner decides on all
                              of a profile's photos, not the first).  Frames with no photo card
                              raise (``NoPhotoCards``): on Hinge that is an extraction failure,
                              so the decision stops rather than scoring a zero block.
``Decision.embedding`` is the v1 vector in both modes; it is what labels and archived passes
store.  The whole-photo part travels separately so the worker can write its own row.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

from ..perception.capture import Profile
from .threshold import (
    DEFAULT_FEATURE_SET,
    FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1,
    FEATURE_SET_INCOMPLETE_KIND,
    FeatureSetIncomplete,
    fetch_label_rows,
    fetch_live_review,
    require_feature_set,
)


@dataclass
class Decision:
    decision: str  # like | dislike | no_face | defer
    score: float = 0.0
    embedding: list[float] = field(default_factory=list)  # empty -> no label stored
    source: str = "ranker"
    # Set only when decide() computed the whole-photo embedding (face_crop_whole_photo_v1
    # mode, and only for a scored like/dislike).  ``frames_sha256`` is the marker: it is the
    # empty string exactly when nothing was computed, and otherwise ``photo_embedding`` holds
    # the 768 floats and ``photo_card_count`` is at least 1.
    photo_embedding: list[float] | None = None
    photo_card_count: int = 0
    frames_sha256: str = ""


class Decider(Protocol):
    def decide(self, profile: Profile) -> Decision: ...


def format_photo_coverage(coverage: Mapping) -> str:
    """``photo_feature_coverage()`` as one readout phrase (``stats`` and the hub card)."""
    return (
        f"whole-photo rows for {coverage['with_features']} of {coverage['labels']} labels "
        f"({coverage['missing']} missing, {coverage['zero_cards']} with zero photo cards)"
    )


def mark_feature_set_incomplete(model, exc: FeatureSetIncomplete) -> bool:
    """Publish ``model`` as NOT READY with ``exc.reason``; returns False (the new ``ready``).

    The one call every loader site (retrain, startup, ``stats``, the hub) makes, so each says
    the same sentence and none keeps serving a classifier fitted on an older label set.  The
    model's ``not_ready_kind`` becomes ``FEATURE_SET_INCOMPLETE_KIND``, which is how an AUTO
    worker that defers tells this backfill problem from a cold start.
    """
    return model.mark_not_ready(
        exc.reason, n_labels=exc.labels, kind=FEATURE_SET_INCOMPLETE_KIND
    )


def whole_photo_block(whole) -> list[float]:
    """The 768-d whole-photo part of a scoring row, from a ``WholePhotoEmbedding``.

    Anything that is not exactly ``WHOLE_PHOTO_DIMS`` numbers raises, a missing vector
    included: there is no zero block.  A Hinge capture with no photo card is an extraction
    failure (``Embedder.embed_whole_photos`` raises ``NoPhotoCards`` for it), and scoring the
    v1 vector plus zeros would be the v1 model in disguise.  A malformed vector must stop the
    decision, never be padded or truncated into a plausible-looking one.
    """
    from ..vision.photo_cards import WHOLE_PHOTO_DIMS

    vector = getattr(whole, "vector", None)
    if vector is None:
        raise ValueError(
            "whole-photo embedding has no vector "
            f"(card_count={getattr(whole, 'card_count', None)!r}); zero photo cards is an "
            "extraction failure, never scored"
        )
    block = list(vector)
    if len(block) != WHOLE_PHOTO_DIMS:
        raise ValueError(
            f"whole-photo embedding has {len(block)} dims; expected {WHOLE_PHOTO_DIMS}"
        )
    return block


class RankerDecider:
    """Composes the quality filter, the embedder, and the PreferenceModel."""

    # How long the live auto-review stats may be reused across retrains.  Training mode retrains
    # after EVERY label; the stats only move when the owner answers a review card or AUTO passes
    # someone, and on BigQuery each read is a query.  A minute-old false-pass estimate moves the
    # recall target by a negligible amount (it is a 10+ sample Wilson bound over a 200-answer
    # window), so a short reuse window removes a per-label round trip at no accuracy cost.
    LIVE_REVIEW_TTL_S = 60.0

    def __init__(
        self,
        quality,
        embedder,
        model,
        *,
        clock: Callable[[], float] = time.monotonic,
        feature_set: str = DEFAULT_FEATURE_SET,
    ):
        self.quality = quality
        self.embedder = embedder
        self.model = model
        self.feature_set = require_feature_set(feature_set)
        self._clock = clock
        self._live_review_cache: tuple[int, float, dict] | None = None

    @property
    def uses_whole_photo(self) -> bool:
        return self.feature_set == FEATURE_SET_FACE_CROP_WHOLE_PHOTO_V1

    def embed(self, profile: Profile) -> list[float] | None:
        """Quality-filter + embed a profile into a feature vector (no scoring).

        Used in observe mode to turn your manual swipe into a training label.
        None when no face is detected.
        """
        photos = self.quality.filter(profile.photos)
        if not photos:  # never drop the whole profile on the filter
            photos = profile.photos
        return self.embedder.embed_profile(replace(profile, photos=photos))

    def embed_whole_photos(self, profile: Profile):
        """The profile's ``WholePhotoEmbedding`` from ALL its frames (no quality filter).

        The same function the backfill tool runs on the stored frames, so a live row and a
        backfilled row of the same capture are identical.  Raises when the embedder cannot
        produce one; callers decide whether that is fatal (see the worker).
        """
        embed = getattr(self.embedder, "embed_whole_photos", None)
        if not callable(embed):
            raise RuntimeError(
                f"{type(self.embedder).__name__} has no embed_whole_photos; the whole-photo "
                "feature cannot be computed"
            )
        return embed(profile.photos)

    def retrain(self, store) -> bool:
        """Reload all labels and retrain the shared model in place. Returns ready.

        Rows carry each label's source (``load_label_rows``) when the store has it, so a
        recall-target model can tell manual calibration labels from answers to auto-pass
        reviews; live review stats feed the model's false-pass correction.  See
        ``threshold.fetch_live_review`` for the best-effort contract of the latter.  A model
        with no recall target never reads them, so the (possibly remote) stats query is skipped.

        Rows are built for ``feature_set``.  When labels lack whole-photo rows the model is
        left NOT READY naming the count and the backfill command (never trained on a subset).
        """
        try:
            rows = fetch_label_rows(store, feature_set=self.feature_set)
        except FeatureSetIncomplete as exc:
            return mark_feature_set_incomplete(self.model, exc)
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
        if not self.uses_whole_photo:
            decision, score = self.model.decide(vec)
            return Decision(decision, score, vec, "ranker")
        # Whole-photo mode: a failure here propagates and stops the run; scoring the v1 vector
        # alone would be a different model than the one that was calibrated.
        whole = self.embed_whole_photos(profile)
        block = whole_photo_block(whole)
        decision, score = self.model.decide(list(vec) + block)
        return Decision(
            decision,
            score,
            vec,
            "ranker",
            photo_embedding=block,
            photo_card_count=int(whole.card_count),
            frames_sha256=str(whole.frames_sha256),
        )
