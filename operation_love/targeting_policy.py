"""Cross-layer readiness for Hinge's numbered still-photo targeting policy.

Keep this module dependency-free so configuration, drivers, and offline tools can all consume the
same fail-closed fact.  The current vision signals can reject visible/high-motion videos, but none
positively distinguishes a still photo from a paused or low-motion video.
"""
from __future__ import annotations


HINGE_PHOTO_SELECTION_POLICY_ID = "hinge_photos_only_v1"
HINGE_POSITIVE_STILL_PHOTO_DISCRIMINATOR_READY = False
HINGE_TARGETING_UNAVAILABLE_REASON = (
    "positive still-photo discriminator unavailable: photographic pixels, low signature drift, "
    "and an absent auto-hiding mute control do not exclude a paused/static video"
)


def hinge_targeting_unavailable_reason() -> str | None:
    """Return the actionable policy blocker, or ``None`` only after positive proof exists."""
    if HINGE_POSITIVE_STILL_PHOTO_DISCRIMINATOR_READY:
        return None
    return HINGE_TARGETING_UNAVAILABLE_REASON
