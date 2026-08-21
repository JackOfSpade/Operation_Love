"""Bumble's staged, currently unlicensed Android binding.

The future Auto path uses direct card-body drags, whose pure mechanics remain unit-testable.
No real session may start while ``BUMBLE_SPEC.calibrated`` and ``observe_ready`` are false;
the paid-upsell detection template is intentionally absent until it is measured live.
"""
from __future__ import annotations

from ..android_spec import AndroidAppSpec
from ..hinge import AndroidDriver
from ...perception.capture import Profile


BUMBLE_SPEC = AndroidAppSpec(
    app="bumble",
    package="com.bumble.app",
    # drivers/android/__init__.py derives Auto and Observe readiness from this spec.  False
    # therefore keeps BOTH modes out of production even though card-drag logic has unit tests.
    calibrated=False,
    # Bumble has no reviewed manual-observation path. Keep this explicit so a future Auto
    # calibration cannot accidentally license Observe merely by changing `calibrated`.
    observe_calibrated=False,
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
    """Bumble's staged direct-card binding; production readiness lives on BUMBLE_SPEC."""

    def __init__(self, cfg):
        super().__init__(cfg, BUMBLE_SPEC)

    def next_profile(self, *, should_stop=None) -> Profile | None:
        """Capture the visible deck card without borrowing Hinge's profile reader.

        The prospective Auto path makes card-body drags; it does not need the Hinge-only
        long-profile scroll, reverse-scroll, item enumeration, or swipe-time opener machinery.
        """
        if should_stop is not None and should_stop():
            return None
        # This override deliberately skips Hinge's long capture implementation, but foreground
        # ownership is app-agnostic. Probe on both sides of the screencap so System UI cannot be
        # returned as a Bumble card if focus changes during the capture command.
        if self._refuse_foreground_block():
            return None
        frame = self._screencap(on_blank="none")
        if frame is None:
            return None
        if self._refuse_foreground_block(frame=frame):
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
