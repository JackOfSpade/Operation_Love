"""Session-scoped interaction policy for autonomous runs.

This module deliberately sits *outside* the preference model.  It shapes when the
already-scored action is made and can conservatively decline a marginal automatic
like, but it never changes training, probabilities, or manual/observe labels.
Everything here is deterministic when supplied a seeded ``random.Random`` and a
local-hour callable, which keeps the behaviour testable without device access.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import math
import random
from typing import Callable

from .human_motion import think_time_s
from .perception.capture import Profile
from .ranker.decider import Decision


_READ_FRACTION_MIN = 0.42
_READ_FRACTION_MAX = 0.60
_READ_X_MIN = 0.42
_READ_X_MAX = 0.58
_READ_DWELL_MIN_S = 0.55
_READ_DWELL_MAX_S = 3.50
_THRESHOLD_MAX_LIFT = 0.045


@dataclass(frozen=True)
class ReadStep:
    """One planned read dwell followed by a forward profile scroll."""

    fraction: float
    x_frac: float
    dwell_s: float


@dataclass(frozen=True)
class PolicyDecision:
    """The policy-adjusted automatic decision plus audit-friendly context."""

    decision: Decision
    effective_threshold: float | None
    demoted: bool = False


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class AutoSessionPolicy:
    """Small, bounded behavioural state for one autonomous worker session.

    ``base_threshold`` is the trained model's configured threshold.  The policy
    only lifts that bar and only when the ranker already returned ``like``.  Thus
    it can never manufacture a positive action from a model ``dislike``.
    """

    def __init__(self, *, rng: random.Random | None = None,
                 local_hour: Callable[[], int] | None = None,
                 base_threshold: float = 0.5,
                 max_threshold_lift: float = _THRESHOLD_MAX_LIFT):
        if not 0.0 < base_threshold < 1.0:
            raise ValueError("base_threshold must be in (0, 1)")
        if not 0.0 <= max_threshold_lift <= _THRESHOLD_MAX_LIFT:
            raise ValueError(
                f"max_threshold_lift must be in [0, {_THRESHOLD_MAX_LIFT}]")
        self.rng = rng if rng is not None else random.Random()
        self._local_hour = local_hour or (lambda: datetime.now().hour)
        self.base_threshold = float(base_threshold)
        self.max_threshold_lift = float(max_threshold_lift)

        # Individual sessions have a coherent pace/strictness instead of every
        # card being an independent draw from exactly the same distribution.
        self._tempo = self.rng.uniform(0.86, 1.16)
        self._read_bias = self.rng.uniform(-0.035, 0.035)
        self._lane_bias = self.rng.uniform(-0.035, 0.035)
        self._strictness = self.rng.uniform(0.0, 0.006)
        self._delay_noise = 0.0
        self.actions = 0
        self.likes = 0
        self._last_action: str | None = None
        self._streak = 0

    @property
    def action_streak(self) -> int:
        return self._streak

    def read_step(self, depth: int, *, captured_frames: int = 1) -> ReadStep:
        """Plan a bounded, safely reversible read scroll.

        ``depth`` and ``captured_frames`` are observations from the current card,
        not model features.  The caller retains the delivered geometry so its
        screenshot-guided return-to-top routine has an accurate travel ledger.
        """
        depth = max(0, int(depth))
        captured_frames = max(1, int(captured_frames))
        complexity = min(1.0, (captured_frames - 1) / 7.0)
        # Deeper cards get slightly shorter advances, avoiding a mechanically
        # identical 55%-of-screen sequence while still remaining in a safe range.
        fraction = _clamp(
            0.51 + self._read_bias - 0.010 * min(depth, 5) + self.rng.gauss(0.0, 0.025),
            _READ_FRACTION_MIN, _READ_FRACTION_MAX)
        x_frac = _clamp(self._lane_bias + 0.50 + self.rng.gauss(0.0, 0.028),
                         _READ_X_MIN, _READ_X_MAX)
        dwell = 1.10 * self._tempo
        dwell *= 1.0 + 0.055 * complexity + 0.065 * min(depth, 6)
        dwell *= math.exp(self.rng.gauss(0.0, 0.25))
        return ReadStep(fraction, x_frac, _clamp(dwell, _READ_DWELL_MIN_S, _READ_DWELL_MAX_S))

    def capture_limit(self, baseline: int) -> int:
        """Vary only the emergency ceiling, never truncate configured coverage."""
        baseline = max(1, int(baseline))
        return baseline + self.rng.choice((0, 0, 1, 1, 2))

    def post_action_delay_s(self, decision: str, profile: Profile, score: float,
                            *, scale: float = 1.0) -> float:
        """Return a contextual wait after a *landed* automatic action.

        The caller supplies the existing pacing scale (and bypasses this method
        entirely for the documented ``swipe_delay_s == 0`` no-pacing mode).
        """
        if scale < 0:
            raise ValueError("scale must be non-negative")
        bucket = "like" if decision == "like" else "pass"
        base = think_time_s(bucket, rng=self.rng)
        complexity = self._profile_complexity(profile)
        uncertainty = 1.0 - min(1.0, abs(float(score) - self.base_threshold) / 0.25)
        fatigue = min(1.0, self.actions / 35.0)
        streak = min(1.0, self._streak / 6.0)
        # Local time is deliberately a small, bounded modifier. It does not
        # attempt to infer identity or replace the user's learned preferences.
        hour = _clamp(float(self._local_hour()), 0.0, 23.0)
        night = 1.0 if hour < 6.0 else (0.5 if hour < 9.0 or hour >= 22.0 else 0.0)
        self._delay_noise = 0.55 * self._delay_noise + 0.45 * self.rng.gauss(0.0, 0.16)
        multiplier = self._tempo
        multiplier *= 1.0 + 0.15 * complexity + 0.12 * uncertainty
        multiplier *= 1.0 + 0.14 * fatigue + 0.08 * streak + 0.09 * night
        multiplier *= math.exp(self._delay_noise)
        # A meaningful floor survives all modifiers.  The upper clamp prevents a
        # rare long-tail draw from turning a normal run into an apparent hang.
        return _clamp(base * multiplier * scale, 0.35 * scale, 45.0 * max(1.0, scale))

    def apply_decision(self, decision: Decision, profile: Profile) -> PolicyDecision:
        """Conservatively apply contextual uncertainty to an auto decision.

        ``defer`` and ``no_face`` retain identity and semantics.  A model dislike
        is also returned unchanged; this policy is intentionally unable to turn a
        negative classifier result into a like.  Only a marginal model like may
        be demoted to an ordinary dislike.
        """
        if decision.decision not in {"like", "dislike"}:
            return PolicyDecision(decision, None)
        if decision.decision == "dislike":
            return PolicyDecision(decision, self.base_threshold)

        threshold = self._effective_threshold(profile, decision.score)
        if decision.score >= threshold:
            return PolicyDecision(decision, threshold)
        return PolicyDecision(
            replace(decision, decision="dislike", source=f"{decision.source}_contextual"),
            threshold,
            demoted=True,
        )

    def record_landed_action(self, decision: str) -> None:
        """Update state only after the driver has confirmed an action landed."""
        if decision not in {"like", "dislike"}:
            raise ValueError("landed action must be 'like' or 'dislike'")
        self.actions += 1
        if decision == "like":
            self.likes += 1
        if decision == self._last_action:
            self._streak += 1
        else:
            self._last_action, self._streak = decision, 1

    def _effective_threshold(self, profile: Profile, score: float) -> float:
        if self.max_threshold_lift == 0.0:
            return self.base_threshold
        complexity = self._profile_complexity(profile)
        # Only ambiguity near the configured boundary needs a second look.  A
        # confident score has no added strictness beyond the session baseline.
        uncertainty = 1.0 - min(1.0, abs(float(score) - self.base_threshold) / 0.10)
        fatigue = min(1.0, self.actions / 35.0)
        like_streak = min(1.0, self._streak / 5.0) if self._last_action == "like" else 0.0
        hour = _clamp(float(self._local_hour()), 0.0, 23.0)
        night = 1.0 if hour < 6.0 or hour >= 22.0 else 0.0
        # Context determines a bounded *available* lift, while a fresh beta draw
        # determines how much of it this card actually receives.  That avoids
        # replacing the model's fixed 0.5 boundary with one equally fixed higher
        # boundary.  The final floor still guarantees no below-model likes.
        available = 0.006 + self._strictness + 0.014 * uncertainty + 0.010 * fatigue
        available += 0.008 * like_streak + 0.004 * night - 0.004 * complexity
        lift = max(0.0, available * self.rng.betavariate(1.4, 2.4)
                   + self.rng.gauss(0.0, 0.002))
        return _clamp(self.base_threshold + lift,
                      self.base_threshold,
                      self.base_threshold + self.max_threshold_lift)

    @staticmethod
    def _profile_complexity(profile: Profile) -> float:
        photos = len(getattr(profile, "photos", []) or [])
        text = len(profile.text_blob()) if hasattr(profile, "text_blob") else 0
        meta = getattr(profile, "meta", {}) or {}
        capture_frames = int(meta.get("capture_frames", photos) or photos)
        read_scrolls = int(meta.get("read_scrolls", max(0, capture_frames - 1)) or 0)
        # Cap every observed dimension so a malformed meta value cannot generate
        # excessive pacing.  Hinge's present capture path supplies photos; these
        # metadata fields let a driver add reading context without model coupling.
        return _clamp(
            0.45 * min(photos, 8) / 8.0
            + 0.25 * min(capture_frames, 8) / 8.0
            + 0.20 * min(read_scrolls, 7) / 7.0
            + 0.10 * min(text, 400) / 400.0,
            0.0, 1.0,
        )
