"""Bumble's Android Auto binding.

This app uses only direct card-body drags for automated decisions. It has no
manual-observation path.
"""
from __future__ import annotations

from ..android_spec import AndroidAppSpec
from ..hinge import AndroidDriver
from ...perception.capture import Profile


BUMBLE_SPEC = AndroidAppSpec(
    app="bumble",
    package="com.bumble.app",
    # Platform readiness is mode-specific in drivers/android/__init__.py: Auto is enabled,
    # while Observe is deliberately unavailable.
    calibrated=False,
    coords={
        # Card-body drags avoid the lower action row, including paid SuperSwipe.
        "swipe_start": (0.50, 0.55),
        "swipe_like_end": (0.92, 0.52),
        "swipe_pass_end": (0.08, 0.52),
    },
    # Measured from the live 1080x2400 deck. These controls are never automation targets.
    forbidden_zones=(
        (0.02, 0.68, 0.23, 0.83),  # compliment
        (0.76, 0.68, 0.98, 0.83),  # paid SuperSwipe
    ),
    # Measured purchase-sheet overlay; inert until an explicit template is calibrated.
    upsell_dismiss_zone=(0.15, 0.10, 0.85, 0.27),
    like_flow="direct",
    decide_gesture="card_swipe",
    accepts_opener=False,
    think_time_calibrated=False,
    change_threshold=9.0,
)


class BumbleAndroidDriver(AndroidDriver):
    """Bumble's direct-card Auto binding."""

    def __init__(self, cfg):
        super().__init__(cfg, BUMBLE_SPEC)

    def next_profile(self, *, should_stop=None) -> Profile | None:
        """Capture the visible deck card without borrowing Hinge's profile reader.

        Bumble Auto makes card-body drags; it does not need the Hinge-only long-profile
        scroll, reverse-scroll, item enumeration, or swipe-time opener machinery.
        """
        if should_stop is not None and should_stop():
            return None
        frame = self._screencap(on_blank="none")
        if frame is None:
            return None
        self._capture_scrolls = 0
        self._capture_scroll_ledger = []
        self._current_capture_truncated = False
        self._current_sigs = []
        self._invalidate_item_index("bumble has no swipe-time opener items")
        if self._dbg is not None:
            self._dbg.action(
                "capture", before=frame, photos=1, capture_truncated=False,
                items=0, item_context=0, item_translation=[], item_manifest=[],
                items_unavailable="bumble has no swipe-time opener items", ranker_photos=None,
            )
        return Profile(
            photos=[frame], prompts=[],
            meta={"app": self.spec.app, "capture_frames": 1, "read_scrolls": 0,
                  "read_dwell_s_total": 0.0, "capture_truncated": False},
            items_unavailable="bumble has no swipe-time opener items",
        )
