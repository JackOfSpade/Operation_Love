"""Human-like timing (log-normal delays) — no network, pure math."""
import random

from operation_love.human import human_cooldown, human_delay


def test_cooldown_never_below_anchor():
    random.seed(1)
    anchor = 5.0
    for _ in range(5000):
        assert human_cooldown(anchor) >= anchor          # backoffs only ever wait longer


def test_delay_spreads_both_sides_of_anchor():
    random.seed(2)
    anchor = 3.5
    samples = [human_delay(anchor) for _ in range(5000)]
    assert all(s > 0 for s in samples)                   # never negative
    assert any(s < anchor for s in samples)              # sometimes quicker
    assert any(s > anchor for s in samples)              # sometimes slower
    mean = sum(samples) / len(samples)
    assert anchor * 0.8 < mean < anchor * 1.3            # centered near the anchor


def test_zero_anchor_is_zero():
    assert human_delay(0.0) == 0.0
    assert human_cooldown(0.0) == 0.0


def test_sigma_zero_returns_anchor_exactly():
    assert human_delay(4.0, sigma=0.0) == 4.0
    assert human_cooldown(4.0, sigma=0.0) == 4.0
